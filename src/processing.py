import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
def Collect_Data(file_path) :
    data = np.genfromtxt(
    file_path,
    delimiter=",",
    dtype=str,
    skip_header=1
    )
    return data


def Create_Exist_Group(data) :
    return data[data[:,1]=='"Existing Customer"',:]

def Create_Attrited_Group(data) :
    return data[data[:,1]=='"Attrited Customer"',:]

def Seperate_Attribution(data) :
    """
    Return : status, age , gender, dependent, edu, martial, income, card, num_month
    """
    status = np.char.strip(data[:,1], '"') # bo dau nhay kep
    age = data[:,2].astype(int)
    gender= np.char.strip(data[:,3],'"')
    dependent = data[:,4].astype(int)
    edu = np.char.strip(data[:,5],'"')
    martial = np.char.strip(data[:,6],'"')
    income = np.char.strip(data[:,7],'"') # 3 type : Less than $40K , $40K - $60K , Others
    card = np.char.strip(data[:,8],'"') # blue, silver, gold, platinum
    num_month = data[:,9].astype(int)
    return status,age,gender,dependent,edu,martial,income,card,num_month

def Print_Statistics(data) :
    status, age,gender,dependent,edu,martial,income,card,num_month = Seperate_Attribution(data)
    #Số lượng người
    print("danh sách có: ", len(data),"khách hàng")
    #giới tính
    print("Số lượng khách nam: ", len(gender[gender=='M']))
    print("Số lượng khách nữ: ", len(gender[gender=='F']))

    #age range
    print("Độ tuổi khách hàng: ", age.min(),'-',age.max())
    #income
    print("Thông tin thu nhập: ")
    print("  + ít hơn $40K: ",len(income[income=='Less than $40K']))
    print("  + $40K - $60K: ",len(income[income=='$40K - $60K']))
    print("  + Khác: ",len(income[(income!='Less than $40K') & (income != '$40K - $60K')]))


def Calculate_Point_biserial_Correlation(binary_np, continuous_np):
    """
    binary_np: numpy array chứa biến nhị phân (0 và 1)
    continuous_np: numpy array chứa biến liên tục
    Return: hệ số tương quan điểm nhị phân
    """
    n = len(binary_np)
    mean_continuous = np.mean(continuous_np)
    mean_binary = np.mean(binary_np)

    std_continuous = np.std(continuous_np)
    std_binary = np.std(binary_np)

    covariance = np.sum((binary_np - mean_binary) * (continuous_np - mean_continuous)) / n #sigma(xi - mean_x)(yi - mean_y) / n

    point_biserial_correlation = covariance / (std_binary * std_continuous)

    return point_biserial_correlation
def Calculate_Chi_Squared(group_np, categorical_np):
    """
    binary_np: numpy array chứa biến nhị phân (0 và 1)
    categorical_np: numpy array chứa biến phân loại
    Return: giá trị chi-squared, df  = (số nhóm -1)*(số loại -1)
    """
    uni_groups = np.unique(group_np)
    uni_categories = np.unique(categorical_np)
    # Tính df
    df = (len(uni_groups) - 1) * (len(uni_categories) - 1)
    # Tạo bảng tần số quan sát
    freq  = np.zeros((len(uni_groups),len(uni_categories)))
    for i, group in enumerate(uni_groups,start=0):
        for j, category in enumerate(uni_categories,start=0):
            freq[i,j] = np.sum((group_np==group) & (categorical_np==category))
    # Tính tổng hàng và cột
    row_sums = np.sum(freq, axis=1)
    col_sums = np.sum(freq, axis=0)
    total = np.sum(freq) # tổng số mẫu
    # Tính bảng tần số kỳ vọng
    expected = np.zeros_like(freq)
    for i, r in enumerate(row_sums,start=0):
        for j, c in enumerate(col_sums,start=0):
            expected[i,j] = (r*c)/total
    # Tính chi-squared
    chi_squared_table = (freq-expected)**2/expected
    chi_squared = np.sum(chi_squared_table)
    return chi_squared, df

# CAC HAM PREPROCESSING : CHUẢN HÓA DỮ LIỆU DẠNG SỐ VỀ PHÂN PHỐI CHUẨN, CHUẨN HÓA DỮ LIỆU DẠNG CATEGORICAL VỀ ONE-HOT ENCODING, Giảm nhiễu = cách xóa Uknown
# Tat ca deu co nen khong can dung ham duoi

def Remove_Unknown_Categorical_Features(data_np, categorical_np_indices):
    mask = np.ones(data_np.shape[0], dtype=bool)
    for index in categorical_np_indices:
        mask &= (data_np[:, index] != '"Unknown"')
    return data_np[mask]

def Standardize_Continuous_Features(continuous_np):
    scaler = StandardScaler()
    continuous_np_reshaped = continuous_np.reshape(-1, 1)  # Chuyển đổi về dạng 2D
    standardized_np = scaler.fit_transform(continuous_np_reshaped)
    return standardized_np.flatten()  # Trả về dạng 1D

def One_Hot_Encode_Categorical_Features(categorical_np):
    encoder = OneHotEncoder(sparse_output=False)
    categorical_np_reshaped = categorical_np.reshape(-1, 1)  # Chuyển đổi về dạng 2D
    one_hot_encoded_np = encoder.fit_transform(categorical_np_reshaped)
    return one_hot_encoded_np
