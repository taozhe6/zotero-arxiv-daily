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
        self.requests_in_current_minute: int = 0
        self.minute_start_time: float = time.time() # 当前分钟的开始时间
        self.last_used_time: float = 0.0 # 用于独立的RPS控制
class SimpleKeyPool:
    def __init__(self, keys: List[str], max_retries_per_key=3, blacklist_threshold=3, recovery_interval_seconds=300, rpm_limit: int = 10):
        """
        初始化简易密钥池。
        :param keys: 初始的 API 密钥列表。
        :param max_retries_per_key: 单个密钥在被列入黑名单前允许的最大失败次数。
        :param blacklist_threshold: 密钥进入黑名单的失败阈值。
        :param recovery_interval_seconds: 黑名单密钥尝试恢复的间隔时间（秒）。
        :param rpm_limit: 每个 Key 每分钟允许的最大请求数。
        """
        if not keys:
            raise ValueError("Key pool must be initialized with at least one key.")

        self._keys: Dict[str, APIKey] = {k: APIKey(k) for k in keys}
        self._active_keys: List[str] = list(keys) # 活跃密钥列表，用于轮换
        self._blacklisted_keys: List[str] = [] # 黑名单密钥列表

        self.max_retries_per_key = max_retries_per_key
        self.blacklist_threshold = blacklist_threshold
        self.recovery_interval_seconds = recovery_interval_seconds
        self.rpm_limit = rpm_limit
        self._lock = asyncio.Lock() # 用于保护 _active_keys 和 _blacklisted_keys 的并发访问

        logger.info(f"SimpleKeyPool initialized with {len(keys)} keys. Blacklist threshold: {blacklist_threshold}, Recovery interval: {recovery_interval_seconds}s, RPM Limit: {rpm_limit}.")
        
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
        此方法现在会等待，直到找到一个未达到速率限制的 Key。
        """
        async with self._lock:
            # 尝试获取一个可用的 Key，直到成功或所有 Key 都被检查过
            start_time = time.time()
            while True:
            # 寻找一个满足所有条件的Key
            best_key = None
            earliest_available_time = float('inf')

            # 遍历所有活跃的Key，而不是只看第一个
            for key_value in self._active_keys:
                api_key_obj = self._keys[key_value]
                current_time = time.time()

                # 检查并重置分钟计数器
                if current_time - api_key_obj.minute_start_time >= 60:
                    api_key_obj.requests_in_current_minute = 0
                    api_key_obj.minute_start_time = current_time

                # 检查RPM和RPS限制
                is_rpm_ok = api_key_obj.requests_in_current_minute < self.rpm_limit
                time_since_last_use = current_time - api_key_obj.last_used_time
                is_rps_ok = time_since_last_use >= self.request_interval_seconds

                if is_rpm_ok and is_rps_ok:
                    best_key = key_value
                    break  # 找到一个完全可用的Key，立即跳出循环
                
                # 如果Key不可用，计算它最早何时可用，用于后续可能的等待
                if not is_rps_ok:
                    key_available_time = api_key_obj.last_used_time + self.request_interval_seconds
                    earliest_available_time = min(earliest_available_time, key_available_time)

            if best_key:
                # 找到了一个立即可用的Key
                api_key_obj = self._keys[best_key]
                api_key_obj.requests_in_current_minute += 1
                api_key_obj.last_used_time = time.time()
                
                # 为了轮换，将被选中的Key移到队列末尾
                self._active_keys.remove(best_key)
                self._active_keys.append(best_key)

                logger.debug(f"Selected key: {best_key[:8]}... (RPM count: {api_key_obj.requests_in_current_minute}/{self.rpm_limit})")
                return best_key
            
            # 如果没有立即可用的Key，说明所有Key都在RPS冷却中
            wait_time = earliest_available_time - time.time()
            if wait_time > 0:
                logger.debug(f"All active keys are in RPS cooldown. Waiting for {wait_time:.2f} seconds.")
                await asyncio.sleep(wait_time)
            # 等待后，重新开始下一次while循环查找
        # --- End of Intelligent Key Selection Logic ---
               

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
        api_key_obj.requests_in_current_minute = 0 # <-- 新增：恢复时重置 RPM 计数
        api_key_obj.minute_start_time = time.time() # <-- 新增：恢复时重置分钟开始时间
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

