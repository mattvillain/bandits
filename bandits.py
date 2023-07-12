from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt
from itertools import product 
import tensorflow as tf
import os
import imageio
import re
from mpl_toolkits.mplot3d import Axes3D
import yaml


def plot_pairwise_combinations(x1,x2,x3,x4,y, title, zlim):
    combinations = [(x1, x2), (x1, x3), (x1, x4), (x2, x3), (x2, x4), (x3, x4)]
    label_combinations = [('Sepal Length', 'Sepal Width'), ('Sepal Length', 'Petal Length'), ('Sepal Length', 'Petal Width'), ('Sepal Width', 'Petal Length'), ('Sepal Width', 'Petal Width'), ('Petal Length', 'Petal Width')]
    # Plotting
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(title)

    for i, (v1, v2) in enumerate(combinations):
        ax = fig.add_subplot(2, 3, i+1, projection='3d')
        ax.scatter(v1, v2, y, c=y, alpha=0.7)
        ax.set_xlabel(label_combinations[i][0])
        ax.set_ylabel(label_combinations[i][1])
        ax.set_zlabel('y')
        ax.set_zlim3d(0,2*zlim) #5e-06)
    fig.savefig(f'{title}.png')

# Define a custom key function for sorting
def numerical_order(filename):
    # Extract the numerical part from the filename using regular expressions
    #word = 'Epoch'
    number = re.search(r'Epoch (\d+).png', filename)
    
    if number != None: 
        return int(number.group(1))

def all_pics_to_a_gif(experiment_name:str, delete: bool):
    ''' 
    pic_names : the last letters of the string 
    directory : where the files must be added - defaults as the current directory
    ''' 
    # Set directory 
    # Example list of filenames
    directory = os.getcwd()
    filenames : list = [file for file in os.listdir(directory) if file.lower().startswith((f'{experiment_name.lower()} : epoch'))]#, '.jpg', '.jpeg'))]
    image_files : list = sorted(filenames, key=numerical_order) # this uses numerical_order method defined above
   
    # Create a list to store the image frames
    frames = []
    for image_file in image_files: # Read each image file and append it to the frames list
        image_path = os.path.join(directory, image_file)
        frames.append(imageio.imread(image_path))
    os.makedirs(directory +'/'+ experiment_name ,exist_ok=True)
    output_path : str = f'{experiment_name}/{experiment_name}.gif' # Path and filename for the output GIF 

    # Save the frames as a GIF
    imageio.mimsave(output_path, frames, duration=2)  # Adjust the duration as needed (in seconds) # COMM // might need to make the duration adaptive to "epoch"
    
    # delete all files: 
    all_files = os.listdir(directory)
    # Iterate over the files and delete ".png" files

    for file in all_files:
        if file.endswith('.png'):
            file_path = os.path.join(directory, file)
            if delete == True: 
                os.remove(file_path)    
            else: 
                os.rename(file_path, f'{experiment_name}' + '/' + file_path)


    print("GIF created successfully!")
    return     

def compute_reward_lipschitz(model,reward_values, X_train):
    '''
    For a model at a given training step and the data, computes the Lipschitz constant. 
    '''
    indices = [i for i in product(range(len(X_train)), range(len(X_train)))] # uses itertools' product to get cartesian product of indices
    store_lipschitz = np.array([np.sum((reward_values[indices[i][0]] - reward_values[indices[i][1]])**2) /(np.sum((X_train[indices[i][0]] - X_train[indices[i][1]])**2)) 
                            for i in range(len(indices))]) # we make an ndarray of this comprehension 
    store_lipschitz[np.isnan(store_lipschitz)] = 0 # set all nans to zero 
    lipschitz_c = np.max(store_lipschitz) # picks the largest lipschitz constant
    return lipschitz_c

# Compute the gradients of the network parameters
class RewardPlotter(tf.keras.callbacks.Callback):
    def __init__(self, X_train : np.ndarray, y_train: np.ndarray, latency : int, rewards : str, experiment_name: str): 
    # store the variables in the objects
        self.X_train = X_train
        self.y_train = y_train
        self.latency = latency
        self.rewards = rewards # can either be "Loss" or "Gradients"
        self.experiment_name = experiment_name

    # initialise new variables useful later
        self.lipschitz_store = [] # list meant to store lipschitz constants through training epochs
        self.zlim = 0 
        self.epoch_counter = 0

    def get_gradientsum(self): 
        '''
        Returns a list storing all the sum of absolute value of all gradients generated
        '''
        store_grads =[]
        for i in range(len(self.X_train)):
            with tf.GradientTape() as tape:
                tape.watch(self.model.trainable_variables)
                pred = self.model(self.X_train)[i]
            gradients = tape.gradient(pred, self.model.trainable_variables)
            running_sum = 0
            for var, grad in zip(self.model.trainable_variables, gradients):
                #print(var.name, grad)
                running_sum += np.abs(np.sum(grad))

            store_grads += [running_sum]
        return store_grads
    
    def get_loss(self):
        return np.sum((self.model.predict(self.X_train) - self.y_train)**2, 1)
    
    def on_train_batch_begin(self, epoch, logs=None):
        if self.rewards == 'Gradients':
            rewards_values = self.get_gradientsum()
        if self.rewards == 'Loss':
            rewards_values = self.get_loss()
        
        x1,x2,x3,x4 = self.X_train[:,0], self.X_train[:,1], self.X_train[:,2], self.X_train[:,3]
        if epoch == 0: 
            self.zlim = max(rewards_values)
        if epoch % self.latency == 0:
            for y in [rewards_values]:  
                #print('Threeway Combinations')
                #plot_threeway_combinations(x1,x2,x3,x4,y)
                title = f'{self.experiment_name} : Epoch {epoch + len(self.X_train) * self.epoch_counter}'
                plot_pairwise_combinations(x1,x2,x3,x4,y,title, self.zlim)
            self.lipschitz_store+=[compute_reward_lipschitz(self.model, rewards_values, self.X_train)]
    
    def on_epoch_end(self, epoch, logs = None):
        self.epoch_counter +=1
        return
    
    def on_training(self, epoch, logs = None):
        fig = plt.figure()
        lipc = np.array(self.lipschitz_store)
        plt.plot(np.arange(0,len(lipc)), lipc)
        plt.title('Lipschitz Value')
        # Save the figure
        plt.savefig(f'Lipschitz/{self.experiment_name} lipschitz.png')

def run_experiment(config):
    # Load the Iris dataset
    iris = load_iris() # <- sklearn.utils._bunch

    # Split the dataset into features and labels
    X : np.ndarray = iris.data 
    y : np.ndarray = iris.target

    encoder = OneHotEncoder()
    y : np.ndarray = encoder.fit_transform(np.reshape(y, (-1, 1))).toarray()

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config['test_size'], random_state=42) # all np.arrays

    # Standardize the features by scaling them to zero mean and unit variance
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # set random seeds for notebook
    seed_value = config['seed_value']
    tf.random.set_seed(seed_value)
    np.random.seed(seed_value)

    # build the small architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(4,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    # mode compile
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    store_loss = np.sum((model.predict(X_train) - y_train)**2, 1)

    # Train the model
    callback = RewardPlotter(X_train, y_train, config['latency'], config['rewards'], config['experiment_name'])
    model.fit(X_train, y_train, epochs = config['epochs'], batch_size= config['batch_size'], verbose=1, callbacks=[callback])
    store_loss = np.sum((model.predict(X_train) - y_train)**2, 1)
    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print('Test Loss:', loss)
    print('Test Accuracy:', accuracy)


def load_config(file_path):
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

if __name__ =='__main__': 
    try: 
        os.mkdir('Lipschitz')
    except FileExistsError: 
        pass
    experiments = load_config('config.yaml')
    for config in experiments: 
        config['experiment_name'] = config['experiment_name'] + '0'
        for i in range(config['rep_number']):
            config['experiment_name'] = config['experiment_name'][:-1] + f'{i}' 
            run_experiment(config)
            all_pics_to_a_gif(config['experiment_name'], delete = config['delete_pics'])

#store_grads = get_gradientsum(model, X_train)
