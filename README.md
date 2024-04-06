# How to Use Mia

Mia is a mod for Celeste that lets anyone build their own neural networks to play the game. It is based on imitative learning: you create a dataset by playing, and Mia learns from it.
The Celeste mod loader “[Everest](https://everestapi.github.io/)” is required to use Mia. Make sure to the enab    le the Debug Mode in the mod options to get access to the debug console.

## Step 1: Create Your Own Dataset

First of all, you need to create your dataset:
- Enter any map, open the debug console (by pressing **‹ . ›** or **‹ ~ ›**) and type `record` to begin recording your gameplay.  
- Play your map as you normally would.  
- Execute `record` in the console once again to stop recording and save your gameplay data. Your data is now saved in the `Mia/Saves` directory.  

**Note**: You can repeat these steps as many times as desired to generate multiple data files.

**Warning**: Due to neural networks simplifications, certain key combinations are forbidden. You can find a list of every allowed key combination in the file `allowedkeys.txt`.

## Step 2: Create and Train Neural Networks
Type in `train` in the debug console to open a cmd and get access to new functions to create and train neural networks:

- **Creating a Neural Network**: Use `create(int list)` to create a neural network using the strucure given in argument. Ensure the first number in the list is `400` and the last is `54`, which are required by Mia.
    - **Example**: `create([400,2048,1024,54])` generates a neural network with one input layer (400 neurons), two hidden layers (2048 and 1024 neurons respectively), and one output layer (54 neurons.)

- **Training Your Neural Network**: Use `train(float, int, int)` to start training, by specifying in order the learning rate,the number of iterations of your dataset, and the sample size of the training.
    - **Note**: If you do not know what a learning rate is, start by setting it at `0.001`. If your neural network does not learn very well, you can attempt to change this value, but do it with caution.

## Step 3: Test Your Neural Network

After creating and training your neural network, you can now look at it play:

- **Restart the level** you wish to test (Pause → Restart Chapter).
- Execute ```play``` in the console to start Mia.
- Look at her go!!

**Note**: To stop Mia, press **‹ a ›**. Due to some coding mystery (and even though the code we wrote should clearly not be doing this), Mia must be either dashing or dying in order to stop playing. If Mia gets stuck and neither is happening, you may need to close the game to stop her.
