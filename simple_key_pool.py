# simple_key_pool.py
import asyncio
import random
import time
from typing import List, Dict, Optional
from loguru import logger

class APIKey:
    """表示一个 API 密钥及其状态的简单数据结构。"""
    def __init__(self, key_value: str):
        self.key_value: str = key_value
        self.failure_count: int = 0
        self.is_blacklisted: bool = False
        self.last_used_time: float = 0.0 # 用于记录上次使用时间，辅助轮换
        self.blacklist_time: float = 0.0 # 记录进入黑名单的时间

class SimpleKeyPool:
    def __init__(self, keys: List[str], max_retries_per_key=3, blacklist_threshold=3, recovery_interval_seconds=300):
        """
        初始化简易密钥池。
        :param keys: 初始的 API 密钥列表。
        :param max_retries_per_key: 单个密钥在被列入黑名单前允许的最大失败次数。
        :param blacklist_threshold: 密钥进入黑名单的失败阈值。
        :param recovery_interval_seconds: 黑名单密钥尝试恢复的间隔时间（秒）。
        """
        if not keys:
            raise ValueError("Key pool must be initialized with at least one key.")

        self._keys: Dict[str, APIKey] = {k: APIKey(k) for k in keys}
        self._active_keys: List[str] = list(keys) # 活跃密钥列表，用于轮换
        self._blacklisted_keys: List[str] = [] # 黑名单密钥列表

        self.max_retries_per_key = max_retries_per_key
        self.blacklist_threshold = blacklist_threshold
        self.recovery_interval_seconds = recovery_interval_seconds
        self._lock = asyncio.Lock() # 用于保护 _active_keys 和 _blacklisted_keys 的并发访问

        logger.info(f"SimpleKeyPool initialized with {len(keys)} keys. Blacklist threshold: {blacklist_threshold}, Recovery interval: {recovery_interval_seconds}s.")
        
        # 启动一个后台任务来定期恢复黑名单密钥
        self._recovery_task = asyncio.create_task(self._periodic_recovery())

    async def _periodic_recovery(self):
        """定期检查黑名单密钥并尝试恢复。"""
        while True:
            await asyncio.sleep(self.recovery_interval_seconds)
            logger.debug("Attempting to recover blacklisted keys...")
            await self.recover_blacklisted_keys()

    async def get_key(self) -> Optional[str]:
        """
        原子性地选择并轮换一个可用的 API 密钥。
        如果所有密钥都在黑名单中，则返回 None。
        """
        async with self._lock:
            if not self._active_keys:
                logger.warning("No active keys available in the pool.")
                return None
            
            # 简单的轮换策略：从列表中取第一个，然后移到末尾
            # 也可以实现更复杂的策略，例如基于上次使用时间或随机选择
            key_value = self._active_keys.pop(0)
            self._active_keys.append(key_value)
            
            api_key_obj = self._keys[key_value]
            api_key_obj.last_used_time = time.time() # 更新上次使用时间
            
            logger.debug(f"Selected key: {key_value[:8]}... (failure count: {api_key_obj.failure_count})")
            return key_value

    async def update_key_status(self, key_value: str, is_success: bool, error_message: Optional[str] = None):
        """
        异步提交密钥状态更新。
        :param key_value: 密钥值。
        :param is_success: 操作是否成功。
        :param error_message: 失败时的错误信息。
        """
        async with self._lock:
            api_key_obj = self._keys.get(key_value)
            if not api_key_obj:
                logger.warning(f"Attempted to update status for unknown key: {key_value[:8]}...")
                return

            if is_success:
                if api_key_obj.is_blacklisted:
                    # 如果密钥在黑名单中但成功了，尝试恢复
                    logger.info(f"Blacklisted key {key_value[:8]}... succeeded, attempting to recover.")
                    await self._recover_key_internal(key_value)
                api_key_obj.failure_count = 0 # 成功则重置失败计数
                logger.debug(f"Key {key_value[:8]}... succeeded. Failure count reset.")
            else:
                api_key_obj.failure_count += 1
                logger.warning(f"Key {key_value[:8]}... failed. Current failure count: {api_key_obj.failure_count}. Error: {error_message}")
                
                if api_key_obj.failure_count >= self.blacklist_threshold and not api_key_obj.is_blacklisted:
                    await self._blacklist_key_internal(key_value)

    async def _blacklist_key_internal(self, key_value: str):
        """将密钥加入黑名单的内部逻辑（不加锁）。"""
        api_key_obj = self._keys[key_value]
        api_key_obj.is_blacklisted = True
        api_key_obj.blacklist_time = time.time()
        
        if key_value in self._active_keys:
            self._active_keys.remove(key_value)
        if key_value not in self._blacklisted_keys:
            self._blacklisted_keys.append(key_value)
        
        logger.error(f"Key {key_value[:8]}... blacklisted due to {api_key_obj.failure_count} failures.")

    async def _recover_key_internal(self, key_value: str):
        """从黑名单中恢复密钥的内部逻辑（不加锁）。"""
        api_key_obj = self._keys[key_value]
        api_key_obj.is_blacklisted = False
        api_key_obj.failure_count = 0
        api_key_obj.blacklist_time = 0.0 # 重置黑名单时间

        if key_value in self._blacklisted_keys:
            self._blacklisted_keys.remove(key_value)
        if key_value not in self._active_keys:
            self._active_keys.append(key_value)
            # 恢复的密钥可以放到活跃列表的开头，给予优先使用
            # self._active_keys.insert(0, key_value) 
        
        logger.info(f"Key {key_value[:8]}... recovered and moved back to active pool.")

    async def recover_blacklisted_keys(self):
        """尝试恢复所有在黑名单中的密钥。"""
        async with self._lock:
            keys_to_recover = []
            current_time = time.time()
            for key_value in self._blacklisted_keys:
                api_key_obj = self._keys[key_value]
                # 简单的恢复策略：达到恢复间隔时间就尝试恢复
                if current_time - api_key_obj.blacklist_time >= self.recovery_interval_seconds:
                    keys_to_recover.append(key_value)
            
            for key_value in keys_to_recover:
                await self._recover_key_internal(key_value)
            
            if keys_to_recover:
                logger.info(f"Attempted to recover {len(keys_to_recover)} blacklisted keys.")
            else:
                logger.debug("No blacklisted keys to recover at this time.")

    async def close(self):
        """关闭密钥池，停止后台任务。"""
        if self._recovery_task:
            self._recovery_task.cancel()
            try:
                await self._recovery_task
            except asyncio.CancelledError:
                logger.info("Key pool recovery task cancelled.")
        logger.info("SimpleKeyPool closed.")

# 示例用法 (仅用于测试 simple_key_pool.py 自身)
async def main_key_pool_test():
    keys = [f"sk-key{i}" for i in range(5)]
    pool = SimpleKeyPool(keys, blacklist_threshold=2, recovery_interval_seconds=10)

    for _ in range(10):
        key = await pool.get_key()
        if key:
            logger.info(f"Using key: {key[:8]}...")
            if random.random() < 0.3: # 模拟 30% 失败率
                await pool.update_key_status(key, False, "Simulated failure")
            else:
                await pool.update_key_status(key, True)
        else:
            logger.warning("No key available, waiting...")
        await asyncio.sleep(1) # 模拟 API 调用时间

    await pool.close()

if __name__ == "__main__":
    asyncio.run(main_key_pool_test())

