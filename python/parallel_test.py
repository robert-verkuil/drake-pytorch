import multiprocessing
from multiprocessing import Pool

def dummy():
    print("dummy")

thingy = "hi"
def f(inp):
    i, x = inp
    print(i, x, thingy)
    x()
    #return x[0]*x[0]

if __name__ == '__main__':
    p = Pool(multiprocessing.cpu_count() - 1)
    # ic_list = [(0, 1), (2, 3), (4, 5)]
    ic_list = [dummy, dummy]
    result = p.map(f, enumerate(ic_list))
    print("about to print result")
    print(result)

