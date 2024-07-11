
import os
import sys
import time
import random
from multiprocessing import shared_memory, Semaphore

MEM_BLOCK_NAME = "mem_block"
MEM_BLOCK_SIZE = 64


def do_child_work(sem, mode):
	shm_c = shared_memory.SharedMemory(MEM_BLOCK_NAME, False, MEM_BLOCK_SIZE)
	for i in range(1000):
		start_acq = time.time()
		sem.acquire()
		end_acq = time.time()
		print("proc {} took {} to acquire the sem".format(os.getpid(), end_acq - start_acq))
		if mode == 'r':
			start_op = time.time()
			read_bytes = bytes(shm_c._buf)
			end_op = time.time()
			print("proc {} took {} to perform the read".format(os.getpid(), end_op - start_op))
			print("The proc read {}".format(read_bytes[0]))
		elif mode == 'w':
			to_write = random.getrandbits(4)
			print("Will write {}".format(to_write))
			bytes_to_write = bytearray([to_write for i in range(MEM_BLOCK_SIZE)])
			start_op = time.time()
			shm_c._buf[:MEM_BLOCK_SIZE] = bytes_to_write
			end_op = time.time()
			print("proc {} took {} to perform the write".format(os.getpid(), end_op - start_op))

		sem.release()
	sys.exit(0)

if __name__ == '__main__':
	print("Py DB")
	sem_main = Semaphore()
	shm = shared_memory.SharedMemory(MEM_BLOCK_NAME, True, MEM_BLOCK_SIZE)
	c1 = os.fork()
	c2 = os.fork()
	if c1 == 0:
		do_child_work(sem_main, 'r')
	elif c2 == 0:
		do_child_work(sem_main, 'w')
	else:
		print(os.waitpid(c1, 0))
		print(os.waitpid(c2, 0))

	print("Processes finished!")
	shm.unlink()
	