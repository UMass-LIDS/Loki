import time
import asyncio

# async def say_after(delay, what):
#     await asyncio.sleep(delay)
#     print(what)

# async def main():
#     print(f"started at {time.strftime('%X')}")

#     await say_after(1, 'hello')
#     await say_after(0.5, 'world')

#     print(f"finished at {time.strftime('%X')}")

# asyncio.run(main())

async def main():
    print('started')
    await asyncio.gather(
        foo(),
        foo(),
    )

async def foo():
    print('starting foo')
    await asyncio.sleep(0.1)
    print('foo finished.')

asyncio.run(main())
