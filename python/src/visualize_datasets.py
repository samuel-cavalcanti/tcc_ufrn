import datasets
from matplotlib import pyplot as plt
import numpy as np




def view_error_t()->None:
    time_k,y_k = datasets.load_error_t_dataset(path=datasets.ERROR_T_KINEMATIC_PATH)
    time_l,y_l = datasets.load_error_t_dataset(path=datasets.ERROR_T_LINEAR_PATH)
    time_a,y_a = datasets.load_error_t_dataset(path=datasets.ERROR_T_ANALYTIC_PATH)

    distance_k = y_k[:,0]
    distance_l = y_l[:,0]
    distance_a = y_a[:,0]

    angle_k = y_k[:,1]
    angle_l = y_l[:,1]
    angle_a = y_a[:,1]

    print(f'distance k: {distance_k[:10]} distance l: {distance_l[:10]} distance a: {distance_a[:10]}')

    

    plt.figure(1)
    plt.plot(time_k,distance_k,label='Kinematic regression',c='r')
    plt.plot(time_l,distance_l,label='Linear regression',c='y')
    plt.plot(time_a,distance_a,label='analytic model',c='g')
    plt.xlabel('segundos')
    plt.ylabel('metros')
    plt.legend()
    plt.savefig(f'plots/distance_over_time_4.pdf')
    plt.figure(2)
    plt.plot(time_k,angle_k,label='Kinematic regression',c='r')
    plt.plot(time_l,angle_l,label='Linear regression',c='y')
    plt.plot(time_a,angle_a,label='analytic model',c='g')
    plt.xlabel('segundos')
    plt.ylabel('radianos')
    plt.legend()
    
    plt.savefig(f'plots/angle_over_time_4.pdf')
    plt.show()


def view_eval_dataset()->None:
    sim_t1,sim_t2 =datasets.load_sim_data_dataset()

    _,y_train = datasets.pre_processing_sim_data(sim_t1,sim_t2)

    _,y_test = datasets.load_sim_5_dataset()
    plot_name = 'conj_dados'
    
    plt.scatter(y_test[:,0],y_test[:,1],label='Conjunto de dados de teste')
    plt.scatter(y_train[:,0],y_train[:,1],label='Conjunto de dados de treinamento e avaliação')
        
    plt.legend()
    plt.savefig(f'plots/{plot_name}.pdf')
    plt.show()

def main()->None:
    view_error_t()
   
   
   


if __name__ == '__main__':
    # test_to_kinematic_data()
    main()


