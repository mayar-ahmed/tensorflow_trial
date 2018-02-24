import numpy as np

def load_dataset(data_dir):
    x_train=np.load(data_dir+'x_train.npy')
    y_train=np.load(data_dir+'y_train.npy')

    x_val=np.load(data_dir+'x_val.npy')
    y_val=np.load(data_dir+'y_val.npy')


    x_test=np.load(data_dir+'x_test.npy')
    y_test=np.load(data_dir+'y_test.npy')

    mean=np.mean(x_train,axis=0)
    std=np.std(x_train,axis=0)

    x_train-=mean
    x_val-=mean
    x_test-=mean

    x_train/=std
    x_val/=std
    x_test/=std

    return (x_train,y_train, x_val, y_val,x_test, y_test)


