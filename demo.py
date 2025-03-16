import matplotlib.pyplot as plt
import time
from fw_diag import fw_diag
from para_adamw import para_adamw
from lovasz import lovasz

if __name__ == '__main__':
    file_name = "Facebook"

    k_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    fw_obj_list = []
    fw_time_list = []

    para_obj_list = []
    para_time_list = []
    
    lovasz_obj_list = []
    lovasz_time_list = []
    
    for k in k_list:                
        start = time.time()
        fw_obj = fw_diag(file_name, k, 1, 200) / (k * (k - 1))
        end = time.time()

        fw_obj_list.append(fw_obj)
        fw_time_list.append(end - start)

        start = time.time()
        para_obj = para_adamw(file_name, k, 1, 200, 3) / (k * (k - 1))
        end = time.time()

        para_obj_list.append(para_obj)
        para_time_list.append(end - start)
        
        start = time.time()
        lovasz_obj = lovasz(file_name, k, 2000) / (k * (k - 1))
        end = time.time()
         
        lovasz_obj_list.append(lovasz_obj)
        lovasz_time_list.append(end - start)
            
    plt.figure(0)
    plt.plot(k_list, fw_obj_list, label = "FW")
    plt.plot(k_list, para_obj_list, label = "AdamW")
    plt.plot(k_list, lovasz_obj_list, label = "L-ADMM")
    plt.xlabel("$k$")
    plt.ylabel("Normalized Edge Density")
    plt.legend()
    plt.xscale("log")
    plt.show()

    plt.figure(1)
    plt.plot(k_list, fw_time_list, label = "FW")
    plt.plot(k_list, para_time_list, label = "AdamW")
    plt.plot(k_list, lovasz_time_list, label = "L-ADMM")
    plt.xlabel("$k$")
    plt.ylabel("Runtime (s)")
    plt.legend()
    plt.xscale("log")
    plt.show()