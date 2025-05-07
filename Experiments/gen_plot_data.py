from calc_time import calc_time
import os

def compile(file, opts, outfile="a.out"):
    os.system(f"nvcc {file} {opts} -o {outfile}")

FOLDER = "./bin"
data_outfile = "plot_data.csv"

f = open(data_outfile, "w")

def compile_all():
    for i in range(1, 33):
        # 1 FP64 and other FP32
        opts = f"-gencode arch=compute_80,code=sm_80 -D NUM_OPS={i}"
        outfile = f"{FOLDER}/fp32_fp64_{i}"
        print(f"Compiling... opts={opts} > {outfile}")
        compile("./fp32_fp64.cu", opts, outfile)

        # all FP 32
        opts += " -D FP32_ONLY"
        outfile = f"{FOLDER}/fp32_{i}"
        print(f"Compiling... opts={opts} > {outfile}")
        compile("./fp32_fp64.cu", opts, outfile)

def run_all(tries = 10):
    for i in range(1, 33):
        outfile = f"{FOLDER}/fp32_fp64_{i}"
        t = calc_time(outfile, 10)/tries
        print(f"Average time (fp32_fp64_{i}): ", t)
        f.write(f"{i},{t},")

        outfile = f"{FOLDER}/fp32_{i}"
        t = calc_time(outfile, 10)/tries
        print(f"Average time (fp32_{i}): ", t)
        f.write(f"{t}\n")
        f.flush()

if __name__ == "__main__":
    # compile_all()
    # print("All compiled")
    
    run_all()
