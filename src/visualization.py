import numpy as np
import matplotlib.pyplot as plt

def Plot_Age_Histogram(age_np) :
    plt.figure()
    plt.hist(age_np,bins=10,edgecolor='black')
    plt.xlabel('Tuổi')
    plt.ylabel('Số Lượng')
    plt.show()
def Plot_Gender_Bar(gender_np):
    plt.figure()
    plt.bar(['Nam','Nữ'],[len(gender_np[gender_np=='M']),len(gender_np[gender_np=='F'])],width=[0.3,0.3],edgecolor='black')
def Plot_Income_Bar(income_np):
    plt.figure()
    plt.bar(['< $40K','$40K-$60K','Khác'],[len(income_np[income_np=='Less than $40K']),len(income_np[income_np=='$40K - $60K']),len(income_np[(income_np!='Less than $40K') & (income_np != '$40K - $60K')])])
    plt.xlabel('Mức thu nhập')
    plt.ylabel('Số lượng khách hàng')
    plt.show()

def Compare_Gender_Each_Group(gender_np1,gender_np2):
    plt.figure()
    exist_counts = [len(gender_np1[gender_np1=='M']), len(gender_np1[gender_np1=='F'])]
    attrited_counts = [len(gender_np2[gender_np2=='M']), len(gender_np2[gender_np2=='F'])]

    labels = ['Nam', 'Nữ']
    colors = ['skyblue', 'pink']

    # Vẽ 2 pie chart cạnh nhau
    fig, axes = plt.subplots(1, 2, figsize=(10,5))

    axes[0].pie(exist_counts, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    axes[0].set_title('Existing Customers')

    axes[1].pie(attrited_counts, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    axes[1].set_title('Attrited Customers')

    plt.show()

def Compare_Income_Each_Group(income_np1,income_np2):
    plt.figure()
    exist_counts = [len(income_np1[income_np1=='Less than $40K']),
                    len(income_np1[income_np1=='$40K - $60K']),
                    len(income_np1[(income_np1!='Less than $40K') & (income_np1 != '$40K - $60K')])]

    attrited_counts = [len(income_np2[income_np2=='Less than $40K']),
                       len(income_np2[income_np2=='$40K - $60K']),
                       len(income_np2[(income_np2!='Less than $40K') & (income_np2 != '$40K - $60K')])]

    labels = ['Thấp hơn $40K','$40K - $60K', 'Khác']
    colors = ['skyblue', 'yellow','green']
    #Vẽ
    fig,ax = plt.subplots(1,2,figsize=(10,5))
    ax[0].pie(exist_counts, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    ax[0].set_title('Existing Customers')

    ax[1].pie(attrited_counts, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    ax[1].set_title('Attrited Customers')
    plt.show()

def Compare_Age_Each_Group(age_np1,age_np2):
    plt.figure()
    bins = range(20, 80, 5)  # Định nghĩa các bin từ 20 đến 80 với bước nhảy là 5

    plt.hist(age_np1, bins=bins, alpha=0.5, label='Existing Customers', edgecolor='black')
    plt.hist(age_np2, bins=bins, alpha=0.5, label='Attrited Customers', edgecolor='black')

    plt.xlabel('Tuổi')
    plt.ylabel('Số lượng khách hàng')
    plt.title('So sánh phân phối độ tuổi giữa hai nhóm khách hàng')
    plt.legend()
    plt.show()
