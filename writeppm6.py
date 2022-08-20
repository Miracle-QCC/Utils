def writePPM6(array,height,width,depth,out_path):
    R = array[:,:,0].flatten()
    G = array[:,:,1].flatten()
    B = array[:,:,2].flatten()

    with open(out_path,'w') as f:
        f.write("P6\n")
        f.write("# Created by CVITEK DPU Model\n")
        f.write("{} {}\n".format(width, height))
        f.write("{}\n".format((1 << depth) - 1))

    with open(out_path,'ab') as f:
        for i in range(height * width):
            rh = (int(R[i]) & 0x0000FF00) >> 8
            rl = (int(R[i]) & 0x000000FF) >> 0

            gh = (int(G[i]) & 0x0000FF00) >> 8
            gl = (int(G[i]) & 0x000000FF) >> 0

            bh = (int(B[i]) & 0x0000FF00) >> 8
            bl = (int(B[i]) & 0x000000FF) >> 0

            f.write(rh)
            f.write(rl)

            f.write(gh)
            f.write(gl)

            f.write(bh)
            f.write(bl)