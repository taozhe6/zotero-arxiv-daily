# llm.py
import asyncio
import random
from typing import List, Dict, Optional
from openai import AsyncOpenAI # 导入异步 OpenAI 客户端
from loguru import logger
from time import sleep

# 导入我们新创建的密钥池
from simple_key_pool import SimpleKeyPool, APIKey 

GLOBAL_LLM_CLIENT = None
GLOBAL_KEY_POOL: Optional[SimpleKeyPool] = None # 全局密钥池实例

class LLMClient: # 重命名为 LLMClient 以避免与 LLM 模块混淆
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None, lang: str = "English", key_pool: Optional[SimpleKeyPool] = None):
        self.model = model
        self.lang = lang
        self.key_pool = key_pool # 接收密钥池实例

        if key_pool:
            # 如果提供了密钥池，则不需要单独的 api_key，AsyncOpenAI 客户端将在每次调用时动态获取
            self.client = AsyncOpenAI(api_key="dummy", base_url=base_url) # api_key 设为 dummy，因为会动态替换
            logger.info(f"LLMClient initialized with KeyPool for model: {self.model}, language: {self.lang}")
        elif api_key:
            # 兼容单个 API Key 的情况
            self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
            logger.info(f"LLMClient initialized with single API Key for model: {self.model}, language: {self.lang}")
        else:
            # 兼容本地 LLM (llama_cpp)
            # 注意：llama_cpp 客户端通常是同步的，且不支持密钥池概念
            # 如果使用本地 LLM，则不会使用密钥池和异步特性
            from llama_cpp import Llama # 延迟导入，避免不必要的依赖
            self.client = Llama.from_pretrained(
                repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF",
                filename="qwen2.5-3b-instruct-q4_k_m.gguf",
                n_ctx=5_000,
                n_threads=4, # 本地 LLM 的线程数
                verbose=False,
            )
            logger.info(f"LLMClient initialized with local Llama model: {self.model}, language: {self.lang}")
        
        logger.debug(f"LLMClient instance initialized with language: {self.lang}")

    async def generate_tldr(self, messages: list[dict]) -> str:
        """
        异步生成 TLDR 摘要。
        如果使用密钥池，将自动轮换密钥并处理失败重试。
        """
        if isinstance(self.client, AsyncOpenAI):
            if self.key_pool:
                # 使用密钥池进行异步调用
                max_attempts_per_request = self.key_pool.max_retries_per_key + 1 # 初始尝试 + 重试次数
                
                for attempt in range(max_attempts_per_request):
                    key_value = await self.key_pool.get_key()
                    if not key_value:
                        logger.error("No active API key available from pool. Cannot generate TLDR.")
                        raise Exception("No active API key available.")

                    try:
                        # 动态设置 API Key
                        self.client.api_key = key_value 
                        response = await self.client.chat.completions.create(
                            messages=messages, 
                            temperature=0, 
                            model=self.model
                        )
                        await self.key_pool.update_key_status(key_value, True) # 成功则更新状态
                        return response.choices[0].message.content
                    except Exception as e:
                        logger.error(f"API call with key {key_value[:8]}... failed (attempt {attempt + 1}/{max_attempts_per_request}): {e}")
                        await self.key_pool.update_key_status(key_value, False, str(e)) # 失败则更新状态
                        
                        # 如果是速率限制错误 (429)，并且还有其他活跃密钥，可以立即尝试下一个密钥
                        # 否则，等待一段时间再重试
                        if "429" in str(e) and len(self.key_pool._active_keys) > 0: # 检查是否有其他活跃密钥
                            logger.warning(f"Rate limit hit with key {key_value[:8]}..., trying next key immediately.")
                            continue # 立即尝试下一个密钥
                        
                        # 如果是最后一个尝试，或者没有其他活跃密钥，则抛出异常
                        if attempt == max_attempts_per_request - 1 or not self.key_pool._active_keys:
                            logger.error(f"All attempts failed or no active keys left for this request.")
                            raise
                        
                        # 否则，等待一段时间再重试 (指数退避)
                        base_delay = 10 # 初始等待时间，根据Google API建议调整
                        await asyncio.sleep(base_delay * (2 ** attempt) + random.uniform(0, 1)) # 2^attempt 秒 + 随机抖动
                
                raise Exception("Failed to generate TLDR after multiple attempts with key pool.") # 理论上不会执行到这里
            else:
                # 单个 API Key 的同步调用 (兼容旧逻辑)
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        response = await self.client.chat.completions.create(messages=messages, temperature=0, model=self.model)
                        return response.choices[0].message.content
                    except Exception as e:
                        logger.error(f"Attempt {attempt + 1} failed: {e}")
                        if attempt == max_retries - 1:
                            raise
                        await asyncio.sleep(3) # 固定等待 3 秒
                raise Exception("Failed to generate TLDR after multiple retries with single API key.")
        else:
            # 本地 LLM (llama_cpp) 仍然是同步调用
            response = self.client.create_chat_completion(messages=messages, temperature=0)
            return response["choices"][0]["message"]["content"]

def set_global_llm_client(api_key: str = None, base_url: str = None, model: str = None, lang: str = "English", key_pool: Optional[SimpleKeyPool] = None):
    global GLOBAL_LLM_CLIENT
    GLOBAL_LLM_CLIENT = LLMClient(api_key=api_key, base_url=base_url, model=model, lang=lang, key_pool=key_pool)

def get_llm_client() -> LLMClient:
    if GLOBAL_LLM_CLIENT is None:
        logger.info("No global LLM client found, creating a default one. Use `set_global_llm_client` to set a custom one.")
        set_global_llm_client() # 默认情况下不使用密钥池
    return GLOBAL_LLM_CLIENT

# 关闭密钥池的函数
async def close_global_key_pool():
    global GLOBAL_KEY_POOL
    if GLOBAL_KEY_POOL:
        await GLOBAL_KEY_POOL.close()
        GLOBAL_KEY_POOL = None
        logger.info("Global key pool closed.")
