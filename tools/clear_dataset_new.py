import os
import json
from pathlib import Path



def make_new_dataset(
    train_folder: str,
    test_folder: str,
    new_path: str,
    ):
    train_path = Path(train_folder)
    test_path = Path(test_folder)
    new_path = Path(new_path)
    new_path.mkdir(parents=True, exist_ok=True)
    labeled_images_path= new_path / 'labeled_images'
    labeled_images_path.mkdir(parents=True, exist_ok=True)
    training_folder=Path(new_path) / 'training_folder/'
    file_list_path = training_folder / 'filelists/'
    training_folder.mkdir(parents=True, exist_ok=True)
    file_list_path.mkdir(parents=True, exist_ok=True)
    class_file=open(train_path / 'class_names.txt','r')
    class_names=class_file.read().splitlines()
    
    cwd = str(os.getcwd())
    
    testset = training_folder / 'testset_ratio_0.3' / 'split_0/'
    testset.mkdir(parents=True, exist_ok=True)
    
    '''building filelists & testset'''
    
    for i in range (11):
        num = 0
        bucket_path = file_list_path / str(i)
        split_bucket_path = testset / str(i)
        bucket_path.mkdir(parents=True, exist_ok=True)
        split_bucket_path.mkdir(parents=True, exist_ok=True)
        all = open(bucket_path / 'all.txt','w+')
        train = open(split_bucket_path / 'train.txt','w+')
        test = open(split_bucket_path / 'test.txt','w+')
        train_indices_path = split_bucket_path /'train_indices.json'
        test_indices_path = split_bucket_path /'test_indices.json'    
        
        bucket_labeled_path = labeled_images_path / str(i)
        bucket_labeled_path.mkdir(parents=True, exist_ok=True)
        train_num=[]
        test_num=[]
        for j in range (len(class_names)):
            labeled_folder = bucket_labeled_path / (str(class_names[j]))
            labeled_folder.mkdir(parents=True, exist_ok=True)
            current_folder = train_path / 'labeled_images' / str(i) / (str(class_names[j]))
            current_folder_str=str(current_folder)
            labeled_folder_str=str(labeled_folder)
            for root, dirs, files in os.walk(current_folder):
                for file in files:
                    os.system('ln -s {} {}'.format(cwd+'/'+current_folder_str+'/'+file,cwd+'/'+ labeled_folder_str+'/'+file))
                    all.write('labeled_images'+'/'+str(i)+'/'+(str(class_names[j]))+'/'+file+' '+str(j)+'\n')
                    train.write('labeled_images'+'/'+str(i)+'/'+(str(class_names[j]))+'/'+file+' '+str(j)+'\n')
                    train_num.append(num)
                    num+=1
            # json_str = json.dumps(train_num)
            # with open(train_indices_path, 'w') as json_file:
            #     json_file.write(json_str)
            current_folder = test_path / 'labeled_images' / str(i) / (str(class_names[j]))
            current_folder_str=str(current_folder)

            for root, dirs, files in os.walk(current_folder):
                for file in files:
                    os.system('ln -s {} {}'.format(cwd+'/'+current_folder_str+'/'+file,cwd+'/'+ labeled_folder_str+'/'+file))
                    all.write('labeled_images'+'/'+str(i)+'/'+(str(class_names[j]))+'/'+file+' '+str(j)+'\n')
                    test.write('labeled_images'+'/'+str(i)+'/'+(str(class_names[j]))+'/'+file+' '+str(j)+'\n')
                    test_num.append(num)
                    num+=1
            # json_str = json.dumps(test_num)
            # with open(test_indices_path, 'w') as json_file:
            #     json_file.write(json_str)
                
        json_str = json.dumps(train_num)
        with open(train_indices_path, 'w') as json_file:
            json_file.write(json_str)
        json_str = json.dumps(test_num)
        with open(test_indices_path, 'w') as json_file:
            json_file.write(json_str)
        all.close()
        train.close()
        test.close()
        class_file.close()
    os.system('cp {} {}'.format(str(train_path)+'/class_names.txt',str(new_path)+'/class_names.txt'))
    bucket_indices=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
    bucket_indices_path=training_folder / 'bucket_indices.json'
    json_str = json.dumps(bucket_indices)
    with open(bucket_indices_path, 'w') as json_file:
        json_file.write(json_str)
            
if __name__ == '__main__':
    train_folder = 'dataset/train_image_only'
    test_folder = 'dataset/test'
    
    make_new_dataset(train_folder=train_folder,test_folder=test_folder,new_path='dataset/testing')
