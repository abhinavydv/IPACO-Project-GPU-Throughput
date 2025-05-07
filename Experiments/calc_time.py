from subprocess import Popen, PIPE
import sys

def calc_time(prog, tries, debug=True):
    t = 0

    for i in range(tries):
        if debug:
            print(f"Running {i+1}/{tries}...", end="\r")
        p = Popen(prog, stdout=PIPE, stderr=PIPE)
        out, err = p.communicate()
        t += float(out.decode().split()[0].strip())

    return t


if __name__ == "__main__":
    tries = 10

    prog = "./a.out"
    if len(sys.argv) == 2:
        prog = sys.argv[1]    

    t = calc_time(prog, tries)
    print(f"\nAverage time: {t / tries:.4f} ms")
