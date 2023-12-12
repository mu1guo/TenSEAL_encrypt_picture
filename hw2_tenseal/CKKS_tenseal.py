import tenseal as ts
import numpy as np
import os
import random
import time
from tqdm import tqdm
import torch
from torchvision.io import read_image
from torchvision import transforms

# 设置参数
batch_size = 4
test_times = 3
image_dir = 'picture/raw'
encrypted_dir = 'picture/encrypt'

# TenSEAL 上下文初始化
context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192*4, coeff_mod_bit_sizes=[60, 40, 40, 60])
context.global_scale = 2**40
context.generate_galois_keys()

# 检查设备并设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("当前设备:", device)

# 获取图像路径列表
image_paths = [os.path.join(image_dir, image) for image in os.listdir(image_dir) if image.endswith('.png')]

# 记录测试时间
test_encrypt_times = []
test_query_times = []
test_decrypt_times = []

# 进行测试
for test_index in range(test_times):
    print("\033[94m进行测试: {}/{}\033[0m".format(test_index+1, test_times))
    
    # 随机选择图像
    selected_images = random.sample(image_paths, batch_size)

    # 计算图像的加密时间
    start_time = time.perf_counter()

    # 将图像转换为 PyTorch 张量
    images = [transforms.Resize((64, 64))(read_image(image_path).to(device)) for image_path in selected_images]
    images = [transforms.ToTensor()(transforms.ToPILImage()(image)).float() for image in images]

    # 加密图像
    encrypted_images = [ts.ckks_vector(context, image.view(-1).tolist()) for image in images]

    # 保存加密后的图像
    for i, encrypted_image in enumerate(encrypted_images):
        with open("{}/encrypted_image_{}.pkl".format(encrypted_dir, i), "wb") as f:
            f.write(encrypted_image.serialize())

    # 计算加密时间
    encryption_time = (time.perf_counter() - start_time)
    average_encryption_time = encryption_time / batch_size

    # 进行查询匹配
    my_query = selected_images[1]  #选择第2张图像进行查询
    start_time = time.perf_counter()

    # 加密查询图像
    query_image = transforms.Resize((64, 64))(read_image(my_query).to(device))
    query_image = transforms.ToTensor()(transforms.ToPILImage()(query_image)).float()
    encrypted_query_image = ts.ckks_vector(context, query_image.view(-1).tolist())

    similarity_scores = []

    for i in range(len(encrypted_images)):
        with open("{}/encrypted_image_{}.pkl".format(encrypted_dir, i), "rb") as f:
            encrypted_image = ts.ckks_vector_from(context, f.read())
            dot_product = encrypted_query_image.dot(encrypted_image)
            similarity_score = dot_product.decrypt()[0]
            similarity_scores.append(similarity_score)

    # 归一化 similarity_scores
    max_score = max(similarity_scores)
    min_score = min(similarity_scores)
    normalized_similarity_scores = [(score - min_score) / (max_score - min_score) for score in similarity_scores]
    # 输出匹配分数
    for i, score in enumerate(normalized_similarity_scores):
        print("Normalized Similarity score with encrypted_image_{}.pkl: {:.4f}".format(i, score))

    # 计算查询时间
    query_time = (time.perf_counter() - start_time)
    average_query_time = query_time / batch_size

    # 计算解密时间
    start_time = time.perf_counter()
    with open("{}/encrypted_image_{}.pkl".format(encrypted_dir, 0), "rb") as f:
        encrypted_image = ts.ckks_vector_from(context, f.read())
        
    decrypted_vector = encrypted_image.decrypt()
    decrypted_image = np.array(decrypted_vector).reshape((64, 64, 3)).astype(np.uint8)
    decrypted_time = time.perf_counter() - start_time

    # 打印测试结果
    print("\033[92m测试{}/{} 结果: \n".format(test_index+1, test_times))
    print("平均加密时间: {:.4f} 秒 / 图像".format(average_encryption_time))
    print("平均查询时间: {:.4f} 秒 / 图像".format(average_query_time))
    print("单次解密时间: {:.4f} 秒 / 图像".format(decrypted_time))

    # 记录测试时间
    test_encrypt_times.append(average_encryption_time)
    test_query_times.append(average_query_time)
    test_decrypt_times.append(decrypted_time)

# 计算平均测试时间
average_encryption_time = sum(test_encrypt_times) / len(test_encrypt_times)
average_query_time = sum(test_query_times) / len(test_query_times)
average_decryption_time = sum(test_decrypt_times) / len(test_decrypt_times)

# 打印最终结果
print("最终结果: ")
print("平均加密时间: {:.4f} 秒 / 图像\n".format(average_encryption_time))
print("平均查询时间: {:.4f} 秒 / 图像\n".format(average_query_time))
print("平均解密时间: {:.4f} 秒 / 图像\n".format(average_decryption_time))
