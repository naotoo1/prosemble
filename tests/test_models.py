"""prosemble.models test suite. """

import numpy as np
import prosemble.models as ps


input_data = np.array(
    [
        [1, 2, 3, 4, 1, 3],
        [1, 0, 3, 4, 1, 3],
        [1, 2, 3, 9, 1, 3],
        [6, 2, 5, 4, 1, 3],
        [1, 0, 3, 2, 1, 7],
        [9, 2, 6, 4, 8, 3],
        [2, 2, 1, 4, 7, 3],
        [2, 2, 0, 4, 7, 3],
        [0, 2, 1, 4, 7, 3],
        [2, 2, 3, 4, 7, 3],
        [1, 2, 1, 4, 7, 3],
        [1, 5, 1, 4, 1, 3],
        [3, 1, 1, 2, 6, 1],
        [9, 1, 3, 1, 6, 4],
    ]
)




def test_hcm_model_build():
    model = ps.Kmeans(
    data=input_data,
    c=3,
    num_inter=100,
    epsilon=0.00001,
    ord='fro',
    plot_steps=True
)

def test_afcm_model_build():
    model = ps.AFCM(
    data=input_data,
    c=3,
    num_iter=1000,
    epsilon=0.00001,
    ord='fro',
    m=2,
    a=2,
    b=2,
    k=1,
    set_U_matrix='fcm',
    plot_steps=True
)

def test_bgpc_model_build():
    model = ps.BGPC(
    data=input_data,
    c=3,
    a_f=2,
    b_f=0.5,
    num_iter=100,
    epsilon=0.00001,
    ord='fro',
    set_U_matrix='fcm',
    plot_steps=True
)
    
def test_fcm_model_build():
    model = ps.FCM(
    data=input_data,
    c=3,
    m=2,
    num_iter=100,
    epsilon=0.00001,
    ord='fro',
    set_U_matrix=None,
    plot_steps=True
)


def test_fpcm_model_build():
    model = ps.FPCM(
    data=input_data,
    c=3,
    m=2,
    eta=2,
    num_iter=1000,
    epsilon=0.00001,
    ord='fro',
    set_U_matrix='fcm',
    plot_steps=True
)

def test_ipcm_model_build():
    model= ps.IPCM1(
    data=input_data,
    c=3,
    m_f=2,
    m_p=2,
    k=1,
    num_iter=None,
    epsilon=0.00001,
    ord='fro',
    set_U_matrix='fcm',
    plot_steps=True
)

def test_ipcm_2_model_build():
    model = ps.IPCM2(
    data=input_data,
    c=3,
    m_f=2,
    m_p=2,
    num_iter=None,
    epsilon=0.00001,
    ord='fro',
    set_U_matrix='fcm',
    plot_steps=True
)


def test_kafcm_model_build():
    model = ps.KAFCM(
    data=input_data,
    c=3,
    num_iter=1000,
    epsilon=0.00001,
    ord='fro',
    m=2,
    a=3,
    b=3,
    k=1,
    sigma=1,
    set_centroids='kfcm',
    set_U_matrix='kfcm',
    plot_steps=True
)
    

def test_kfcm_model_build():
    model = ps.KFCM(
    data=input_data,
    c=3, 
    num_iter=1000,
    epsilon=0.001,
    ord='fro',
    set_prototypes=None,
    m=2,
    sigma=1,
    set_U_matrix=None,
    plot_steps=True
)


def test_kpfcm_model_build():
    model = ps.KFPCM(
    data=input_data,
    c=3,
    num_iter=100,
    epsilon=0.0001,
    ord='fro',
    m=2,
    sigma=1,
    eta=2,
    set_centroids=None,
    set_U_matrix='kfcm',
    plot_steps=True
)


def test_kipcm_model_build():
    model = ps.KIPCM(
    data=input_data,
    c=3,
    num_iter=1000,
    epsilon=0.001,
    ord='fro',
    m_f=2,
    m_p=2,
    k=1,
    sigma=10,
    set_centroids='fcm',
    set_U_matrix='fcm',
    plot_steps=True
)



def test_kipcm2_model_build():
    model = ps.KIPCM2(
    data=input_data,
    c=3,
    num_iter=100,
    epsilon=0.0001,
    ord='fro',
    m=2,
    k=2,
    sigma=10,
    set_centroids='fcm',
    set_U_matrix='fcm',
    plot_steps=True
)   

def test_kpcm_model_build():
    model = ps.KPCM(
    data=input_data,
    c=3,
    num_iter=100,
    epsilon=0.001,
    ord='fro',
    m=2,
    k=0.06,
    sigma=1,
    set_centroids=None,
    set_U_matrix='kfcm',
    plot_steps=True
)


def test_pcm_model_build():
    model= ps.PCM(
    data=input_data,
    c=3,
    m=2,
    k=0.001,
    num_iter=1000,
    epsilon=0.00001,
    ord='fro',
    set_U_matrix='fcm',
    plot_steps=True
)

def test_pfcm_model_buidl():
    model= ps.PFCM(
    data=input_data,
    c=3,
    m=2,
    eta=2,
    k=1,
    a=2,
    b=2,
    num_iter=1000,
    epsilon=0.00001,
    ord='fro',
    set_U_matrix='fcm',
    plot_steps=True
)

