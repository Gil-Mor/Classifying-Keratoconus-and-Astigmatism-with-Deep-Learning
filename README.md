# Classifying Keratoconus and Astigmatism with Deep Learning  
#### Final-Year project in Deep learning in collaboration with the Optometry Department in Hadassah College.  
The aim was to diagnose 2 types of Astigmatism in eye images, using deep learning (i.e. no feature engineering).     
The project was done in collaboration with the college Optometry Department.   
I had 220 eye images so it was impossible to train a neural net from scratch for this task.   
Instead, I used Transfer Learning - I took an AlexNet model that was already trained on the ImageNet data set and I   
resumed its learning process on the new data set for the new task.   
After I managed to configure the Hyperparameters to learn from the small data set   
the net achieved 93.6% accuracy in 5 minutes training time.  
Check out the [poster](https://github.com/Gil-Mor/Classifying-Keratoconus-and-Astigmatism-with-Deep-Learning/blob/master/Poster.pdf) and enjoy reading the full [report](https://github.com/Gil-Mor/Classifying-Keratoconus-and-Astigmatism-with-Deep-Learning/blob/master/Report.pdf).

  
### Using EC2 AWS Instance
* I worked on the EC2 instance from both Windows and Linux. 
* When working from Linux everything was done via ssh from the cmdline using bash scripts.  
When working from windows I used putty.  
[ See here ](http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AccessingInstances.html) on how to connect to aws instance from Win or Linux.  

* Every time you activate the instance, the *public dns* changes and you have to replace it the scripts or in Putty (depends if you're connected from Win or Linux).  
 
* I prepared a Jupyer infrastructure for the public presentation day so you can continue with that. Check the section about run_demos.ipynb.


### Installing Caffe from scratch on Ubuntu Desktop or AWS
Caffe2 came out while I was finishing my project, so I don't recommend others to install Caffe version 1. It'll probably be deprecated at some point.

Anyway, this is how I installed it:  
[ install Caffe ](http://caffe.berkeleyvision.org/installation.html) on Ubuntu 14.04 Desktop  
or on [ AWS EC2 GPU instance, e.g., g2.xlarge ](https://github.com/BVLC/caffe/wiki/Install-Caffe-on-EC2-from-scratch-\(Ubuntu,-CUDA-7,-cuDNN-3\)).


Install python2.7 Anaconda distribution.  
To run my code, you'll need the extra packages: opencv and natsort. (You don't need these packages for elementary things, only for some specific tasks)  

install opencv and natsort via conda:  
`$ conda install -c conda-forge opencv`  
`$ conda install -c anaconda natsort`  


### The Code

##### Notes about the Code  
Some experiments we're deliberately counter-productive (mislabeling the images to check for errors in training process,
testing for the minimum required data to train the net on Healthy vs. KC., etc..).  
Those experiments are the cause to some of the mess in the code. If I only had to do straight-forward flows the code would be much cleaner.

There's too much to cover but the general idea is simple. Just follow the flows, reverse engineer, and search the web.  


###### Mode Class
Is a class that defines an experiment: Its name, path, flow, its CNN parameters, is it a repeated experiment or a fresh one, etc...
Almost every flow starts with specifying a `mode` to work with.

In many cases we want to work with an existing mode; To re calculate statistics, re plot graphs etc.., In this case we'll also
declare a Mode instance and use its methods to access the pre-existing configurations and statistics files.

###### Submodes
In cross-validation (CV) we start from a `root_mode` and then for each CV iteration I create a `submode` from that `root_mode`.
Each `submode` has its own unique validation set.

###### modes.py 
Contains various modes I created.

###### export_env_variables.py
Contains various static definitions. most notably - paths.


###### platform_defs.py
export_env_variables.py depends on the platform we're currently running on.
`platform_defs.py`should define a variable called `PLATFORM` which specifies the platform type we're currently using. 
The file is not in the git repository, it should be created locally on the platform.
Here's my `platform_defs.py` from my laptop:

```
EC2                = "ec2"          # Some AWS EC2 Instance  
EC2_GPU_Platform   = "ec2_gpu"      # AWS EC2 GPU instance. Set Caffe to GPU mode  
EC2_CPU_Platform   = "ec2_cpu"      # AWS EC2 CPU instance. Set Caffe to CPU mode. AWS EC2 instances have more RAM than my moderate PC, which affects the maximum batch size.  
PC_Platform        = "pc"           # My Laptop - moderate RAM, CPU only, can no longer run Caffe.  
PLATFORM           = PC_Platform    # Choose your platform  
```

The code is built in a way that you can run Caffe on the server, than copy to output files to you PC and then you can re calculate things or re plot graphs 
using the logs without a need to run the net again.  
Use the PLATFORM variable to control the flows.


###### uber_script.py
The entry point to all experiments.  
Runs various experiments and contains widely used auxiliary methods like gathering and averaging statistics.  
In the file main method I set the `mode` (the experiment specifications) and call the required method.  
For example: 
```
mode = healthy_vs_kc_vs_cly_cross_validation_adam_100_epochs  
train_predict(mode, first_set_i=1, last_set_i=-1, mode_of_operation=CROSS_VALIDATION, from_scratch=False)
```

will Fine-tune caffenet with cross-validation on Healthy vs. KC vs. Astigmatism for a 100 epochs. In the end it'll  
average all the results and plot the learning curves.

##### Prerequisite for training a net in caffe:
1. train/val txts - Contain a list of file names of images and their labels. train.txt contains the list of images for the training set  
and val.txt contains a list of images for the validation set.
1. Two [ lmdb ](https://en.wikipedia.org/wiki/Lightning_Memory-Mapped_Database) files. LMDBs are sequential data bases.  
They contain images for the net. The net pulls a `batch` of images in every iteration.  
The order is sequential and circular. The LMDBs are created from train/val txts with caffe cmdline utils.
1. `.binaryproto` files with images-mean. Created from `train.lmdb` with caffe's cmdline utils.  
1. Nets configuration files (`prototxt` files). For each mode Caffe defines 3 nets: Train-net, Test-net, and Deploy-net.  
    - Train-net and Test-net are used during training.  
      Train-net is used for training during training and test-net is used for validation during training.  
      They're defined in a `prototxt` file that ends with `train_val.prototxt`.  
    - Deploy-net is defined in a `prototxt` file that ends with `deploy.prototxt`. Deploy-Net is used for *real world* prediction of *real* images outside of the training.  
1. To fine-tune a pre-trained model you'll need its existing weights. This file ends with (`.cafemodel`).  
1. To resume the training of a net you already started training you'll need a snapshot of the weights you want to resume from and a corresponding `.solverstate` file that contains the current optimization stage (learning rate etc...).


##### General Train and Predict flow
Check out `train_predict` method in `uber_script.py` file.  
Suppose you want to fine-tune the trained caffenet model and do cross-validation.
```
1. Specify a `root_mode`.  
2. Calculate how many validation sets you'll have, The validation sets and training sets size.  
3. Create a submode for the cross-validation iteration. (called `set_<set_number>`). 
   For each submode:  

    * Split the data to train set and validation set and write the sets to train/val txts.
    * Prepare two LMDBs for training set and validation set.
    * Prepare image-mean from the `train.lmdb`. 
    * Write `prototxt` (configuration) files containing the: 
        * solver configurations (ends with `solver.prototxt`). 
        * train-validation-net configuration (ends with `train_val.prototxt`). 
        * deploy-net configurations (ends with `deploy.prototxt`). 
    * Train the net. Use caffe cmdline interface (called from *run_transfer_learning_caffe_cmd* method) 
    * During Training, the learned weights are automatically saved as *snapshots* every specified number of Iterations. 
    * After the training is over, use the saved *snapshots* to predict the validation set and save the net's predictions and probabilities to txt files (`call_predict_on_train_and_val_txts` method). 
        * You can't do this step during training. You'll have to deploy a deploy-net in any way. 
        * Note that when deploying the net and predicting on real world images, the images have to be reprocessed just like the `train_val` net processes them: center-crop from 256X256 to 227X227 and do mean-subtraction. 
        * During this stage I write the results to txt files (`train/val_classification\_<iteration>.log`)  

4. Average all `train/val_classification_<iteration>` logs and plot graphs (`average_results_plot_averages_plot_by_classes` method). 
```

###### run_demos.ipynb
Streamlined API created for the public-day demonstrations.  
runs on [ Jupyter Notebook ](http://jupyter.org/) to provide a graphical window into the EC2 server.  
   	Runs an Interactive guessing game against a fine-tuned net (Healthy vs. KC vs. Astigmatism).  
	Runs cross-validation on Healthy vs. KC vs. Regular Astigmatism.  
Calls `train_predict` in `uber_script.py` for running the net.
[ see here ](http://docs.aws.amazon.com/mxnet/latest/dg/setup-jupyter-configure-server.html) on how to configure Jupyter on AWS EC2.  
[ and here ](http://docs.aws.amazon.com/mxnet/latest/dg/setup-jupyter-configure-client-windows.html) on how to set up your Windows client.  
I preferred to [ connect from my Ubuntu ](http://docs.aws.amazon.com/mxnet/latest/dg/setup-jupyter-configure-client-linux.html) since it was much easier to set up.  
On my AWS instance, go to `scripts` folder and execute `$ Jupyter notebook` to start the Jupyter kernel.

###### demo_mode variable
Defined in `export_env_variables.py`. If set to `True` it will run a 'demo' version of train_predict.  
It will only predict the validation set (normally I also predict the training set to get more insight).  
It will also not plot predictions by classes. 

#### Manually train the net
You can use _pycaffe_ interface to train the net instead of using the more restrictive cmdline.  
Due to a buggy logging system, Caffe log won't be saved into a file if you're not running from cmdline.   
I worked around this issue by defining the manual training flow in an external file called `solve.py` and called it from uber script - while redirecting its stdout to a file.  
(`call_external_solve` in uber_script).   
I experimented a lot with manual solve but then neglected it so I don't know what's the state of this method but it's a good start if you want to run manually.  

### More Code Files

##### defs.py
Contains classes definitions like:   
1. `Mode` - already explained.  
2. `Solver_Net_parameters` - CNN parameters. Created seperatly from Mode but given to `Mode` as an argument.  
3. `Txts_data` - sort of deprecated. Was used to build train and validation images lists txt files but it's not so much in use since I started using cross-validation which uses a different method for splitting the data to train and validation sets.  

##### utils.py
Contains auxiliary methods. Used extensively.

##### make_train_val_txt.py
Deprecated. Was used in the beginning. `write_train_val_txts` in `utils.py` can be used instead although I didn't use it
recently either.

##### plot_learning_curve.py
Create plots.  

##### visualizations.py  
Contains many experimental methods for visualizing the net's layers.  
Check out `visualize_net_layers` in `uber_script`.   
Note that only the first 2 convolution can make some sense.  
Note that it's also different to visualize the weights them selves than to visualize a filtered image.  

from *visualize_net_layers*   
```
# Extract 9 convolution filters from the first convolution layer.
feat = net.blobs['conv1'].data[0, :9]
vis_square(feat, filename="conv1_layer")
```  
  
vs.  
  
from `interactive_predict_val_txts` in `run_demo.ipynb`  
```
# copy the image data into the memory allocated for the net
net.blobs['data'].data[...] = transformed_image

# propagate the image through the net.
output = net.forward(start='conv1')

# extract 'convolutionized' image and save the image.
feat = net.blobs['conv1'].data[0, :9]
vis_square(feat, filename="convolutionized_image")
```

###### prepare_data_from_pdf.py
Contains methods for preparing jpgs from pdfs and also to organize all patients data to a comprehensive format.
I used it extensively when I got 2 new batches of data during the project. 
The data had no unified format so it was hard to do things automatically.
It wasn't a linear procedure. I don't remember everything that went on there. 
Take it as a reference.

###### preprocessing.py
Contains methods for augmenting data before training.  
I didn't find augmented data helpful so this wasn't used much.  

###### write_prototxts.py
Write `prototxt` (Configuration files) according to `mode`.  

###### save_logs.py
Save all logs of a certain `mode` to a special folder for downloading them from AWS to my PC.  
Can be used by calling `$ python uber_script.py save_logs <mode_name>` from the cmdline 


### Other Files

#### All Images
Contain folders with images

##### full_size_with_info_names
Contain full-size images (straight from Schwind Sirius PDF) with file names containing info from patients_info_updated.xlsx excel.

##### full_size_with_index_names
Contain full-size images (straight from Schwind Sirius PDF) with file names containing only category and index.

##### kc
All images cropped to 256X256.


#### connect_to_aws folder
Bash scripts to connect with the EC2 instance and Jupyter.  
All those scripts were located in my `home` directory on Ubuntu but they all `cd` to `.ssh` directory where my `keys` are.  
If you're connecting from windows you need to use different tools.
* `ec2_dns.sh` - I put the newly generated public dns here every time I start the instance. The script exports the variable to all scripts that call it.
* `connect_to_ec2.sh` - opens a ssh terminal to the instance.
* `download_saved_logs_from_ec2.sh` - uses `rsync` to download folders from the ec2 to the client. `rsync` was much faster than `scp`.
* `jupyter_notebook.sh` - open connection to the Jupyter server on the AWS. (First start the server on AWS). This will open a browser tab.

#### data_info_files folder
* patients_info_updated.xlsx - contains all the patients info and their assigned *index* for the `<category>_<index>` format, e.g. `healthy_1.jpg`.
* kc_sus_kc_probabilities.txt - Contains a list of KC suspects (1-40) and my subjective evaluation of how much each suspect resembles a KC patient.  
0 for "The image look totally healthy" and 1 for "the images looks totally like a KC patient".  
This was used by the `plot_suspects_predictions` in `uber_script`.  
First you have to predict the suspects either by training a mode with suspects or to predict the suspects with snapshots that were saved from a mode that wasn't trained on suspects, i.e., trained on Healthy vs. KC). Put this file in `kc` images folder to use it.

#### ec2_modes_logs folder
Contains logs from experiments on AWS.

#### misclassified_images folder
Contains sub folders with examples of images that were misclassified, e.g. healthy_as_cly.  
This are only some examples of recurrent misclassification and they do not represent all cases.


### Good references
[ caffe examples ](https://github.com/BVLC/caffe/tree/master/examples)  
[ practical guide to caffe, SGD, and CNNs ](https://software.intel.com/en-us/articles/training-and-deploying-deep-learning-networks-with-caffe-optimized-for-intel-architecture)  
[ caffe ](http://caffe.berkeleyvision.org/)  
[ caffe users group - the place to ask technical questions ](https://groups.google.com/forum/#!forum/caffe-users)  


__Gil Mor__


