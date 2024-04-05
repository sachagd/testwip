# How to Use Mia

Mia is a mod for Celeste that lets anyone build their own neural networks to play the game. It's based on imitative learning : you create a dataset by playing, and Mia learns from it.


## Step 1: Create Your Own Dataset

First of all, you need to create your dataset :
- Open any level, then access the command line and execute `record` to start recording your gameplay.  
- Play your map as you normally would.  
- Re-execute `record` in the command line to stop recording and save your gameplay data. Your data is now saved in the `Mia/Saves` directory.  

**Note**: You can repeat these steps as many times as desired to generate multiple data files.

**Warning**: Due to neural network simplification, certain key combinations are not allowed. As a result, you can find a list of every key combinations in the file `allowedkeys.txt`.

## Step 2: Create and Train Neural Networks
Execute `train` in the command line to open a cmd and access new functions for creating and training neural networks:

- **Creating a Neural Network**: Use `create(int list)` to create a neural network of the strucure given in argument. Ensure the first number in the list is `400` and the last is `54`, which are required by Mia.
    - **Example**: `create([400,2048,1024,54])` generates a neural network with one input layer (400 neurons), two hidden layers (2048 and 1024 neurons, respectively), and one output layer (54 neurons).

- **Training Your Neural Network**: Use `train(float, int, int)` to start training, specifying the learning rate, number of iterations over your dataset, and training sample size.
    - **Note** : If you don't at all what is the learning rate, start with `0.001`. If your neural network don't seems to learn, you can try to change this value but do it carefully.

## Step 3: Test Your Neural Network

After creating and training your neural network, you can now test it:

- **Restart the level** you wish to test.
- Execute ```play``` in the command line to start mia.

**Note**: To stop Mia, press **"a"**. However due to some coding mistery and even though the code i wrote should clearly not do that, Mia must be doing a dash or dying to successfully stop using "a". If Mia gets stuck and neither is happening, you may need to close the game to stop it.