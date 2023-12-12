# 引入必要的库
import tensorflow as tf
import tenseal as ts
import numpy as np
import os
import random
import time
from tqdm import tqdm

# 设置参数
batch_size = 4     # 读入图片的数量
test_times = 3     # 测试的次数，每次5张图像
image_dir = 'picture/raw'                # 原图路径
encrypted_dir = 'picture/encrypt'        # 加密后的图像数据的路径

# TenSEAL上下文设置 - BFV方案
context = ts.context(ts.SCHEME_TYPE.BFV, poly_modulus_degree=8192*4, plain_modulus=786433)
context.generate_galois_keys()

# 确保 TensorFlow 使用 GPU
physical_devices = tf.config.list_physical_devices()
print(physical_devices)
print("可用GPU数量: ", len(physical_devices)-1)
if len(physical_devices) > 1:
    tf.config.experimental.set_memory_growth(physical_devices[1], True)

# 获取图像路径列表
image_list = [os.path.join(image_dir, image) for image in os.listdir(image_dir) if image.endswith('.png')]

# 记录每一轮的时间结果--数组
test_encrypt_times = []
test_query_times = []
test_decrypt_times = []

#利用for循环，测试5次咯，不过，主要还是研究for循环中每一次的操作 —— 每次都会对5张图像进行处理
for times in range(test_times):
    #(1)输出这是第几轮测试
    print("\033[94m正在测试: {}/{}\033[0m".format(times+1, test_times))

    #（2）从图像文件夹中的10张.png图像中随机选择5张图像
    random_images = random.sample(image_list, batch_size)

    #（3）利用调用time库中的perf_counter()函数记录开始的时“刻”
    start_time = time.perf_counter()  #start_time里面是开始值
    
    # (4)图像转为 TensorFlow 张量
    images = [tf.io.read_file(image) for image in random_images]           #转化为tf张量
    images = [tf.image.decode_png(image, channels=3) for image in images]  #对png解码
    images = [tf.image.resize(image, [64, 64]) for image in images]        #调整为64*64大小——后期恢复估计会有损

    # (5)加密
    encrypted_images = []                                                           #数组中的元素[i]：图像->转向量->对向量加密
    for image in images:                                                            #利用for循环-处理5张图像
        encrypted_vector = ts.bfv_vector(context, image.numpy().flatten().tolist()) #图像->numpy->flatten展平->tolist变成向量list
        encrypted_images.append(encrypted_vector)                                   #加密后的vector存到数组中

    # (6)将加密图像保存到文件
    for i, encrypted_image in enumerate(encrypted_images):                          #将一个个加密后的vec存到对应的第i个.pkl文件中
        with open("{}/encrypted_image_{}.pkl".format(encrypted_dir, i), "wb") as f: #
            f.write(encrypted_image.serialize())                                    #以将每个加密后的vec->转成序列化格式->write写入保存


    # （7）计算加密时间  ---- 计算的时间包括： 5张图像转tf张量+加密5张图像+5张序列化保存
    # 包括5张图像的时间 和 每张图像的平均时间
    encryption_time = (time.perf_counter() - start_time)  # 转为微秒: * 1e6
    average_encryption_time = encryption_time / batch_size

    # （8）匹配查询
    my_query = random_images[1]  #选择第2张图像进行查询

    start_time = time.perf_counter()     #再次记录开始时“刻”

    # （9）加密查询图像 --  同样的操作
    query_image = tf.io.read_file(my_query) 
    query_image = tf.image.decode_jpeg(query_image, channels=3)
    query_image = tf.image.resize(query_image, [64, 64])
    encrypted_query_image = ts.bfv_vector(context, query_image.numpy().flatten().tolist())

    #（10）比较查询的分数
    similarity_scores = []  #记录分数的数组

    for i in range(len(encrypted_images)):                                            #逐个打开i=1-5的.pkl文件
        with open("{}/encrypted_image_{}.pkl".format(encrypted_dir, i), "rb") as f:   #利用for循环读出来
            encrypted_image = ts.bfv_vector_from(context, f.read())                   #将f文件中读取的内容通过context加密得到encrypt_img
            dot_image = encrypted_query_image.dot(encrypted_image)       #先将查询图像 和 encrypt[i]做的点乘的结果存到sim_score[i]中
            similarity_score = dot_image.decrypt()[0]                    #先解密在存的，所以是一个数值，注意，我觉得还要除以各自的模长
            similarity_scores.append(similarity_score)                   #后面会进行归一化处理，别急

    # 归一化 similarity_scores
    max_score = max(similarity_scores)
    min_score = min(similarity_scores)
    normalized_similarity_scores = [(score - min_score) / (max_score - min_score) for score in similarity_scores]
    # 输出匹配分数
    for i, score in enumerate(normalized_similarity_scores):
        print("Normalized Similarity score with encrypted_image_{}.pkl: {:.4f}".format(i, score))
    
    #（11）计算得到 比较查询需要的时间 - 
    # 包括5张图像的时间 和 每张图像的平均时间
    query_time = (time.perf_counter() - start_time)
    average_query_time = query_time / batch_size
    
    #（12）计算解密一张图像的时间
    start_time = time.perf_counter()
    with open("{}/encrypted_image_{}.pkl".format(encrypted_dir, 0), "rb") as f:  #直接打开0.pkl文件解密出图像
        encrypted_image = ts.bfv_vector_from(context, f.read())
        
    decrypted_vector = encrypted_image.decrypt()
    decrypted_image = np.array(decrypted_vector).reshape((64, 64, 3)).astype(np.uint8)
    
    decrypted_time = time.perf_counter() - start_time

    #（13）输出加密时间，查询时间，单次解密时间
    print("\033[92m测试{}/{} 结果: \n".format(times+1, test_times)) 
    print("平均加密时间: {:.4f} 秒 / 图像".format(average_encryption_time))
    print("平均查询时间: {:.4f} 秒 / 图像".format(average_query_time))
    print("单次解密时间: {:.4f} 秒 / 图像".format(decrypted_time))
    
    #（14）这一轮的结果存到这3个时间数组中
    test_encrypt_times.append(average_encryption_time)
    test_query_times.append(average_query_time)
    test_decrypt_times.append(decrypted_time)

#（15）计算总的平均加密，查询时间,解密时间
average_encryption_time = sum(test_encrypt_times) / len(test_encrypt_times)
average_query_time = sum(test_query_times) / len(test_query_times)
average_decryption_time = sum(test_decrypt_times) / len(test_decrypt_times)

print("最终结果: ")
print("平均加密时间: {:.4f} 秒 / 图像\n".format(average_encryption_time))
print("平均查询时间: {:.4f} 秒 / 图像\n".format(average_query_time))
print("平均解密时间: {:.4f} 秒 / 图像\n".format(average_decryption_time))

