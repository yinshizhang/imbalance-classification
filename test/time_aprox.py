import sys
import time
from train import run

start = time.time()
run(dsname='vehicle', sampling='nonsampling', seed=int(sys.argv[1]), test_size=0.2, epochs=1000)
end = time.time()

print(f"Time elapsed for job {int(sys.argv[1])}: {end - start}")