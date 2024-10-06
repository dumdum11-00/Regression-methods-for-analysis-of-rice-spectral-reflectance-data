
def log(msg):
    f = open("log.txt", "a")

    for line in msg:
        f.write(line)
        f.write("\n")
    f.close()
    