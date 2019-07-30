# manifold_learning

### Getting the code
Clone this repository. From your preferred directory execute:
```
git clone https://github.com/dhudsmith/manifold_learning.git
```

### Docker
Install Docker on your machine. You can find installation instructions for various operating systems on the [Docker website](https://docs.docker.com/install/). 

Once docker is installed, download the docker container image for this project by executing the command below from a terminal (current working directory does not matter). The docker images is very large, so this will take a while. Go make yourself a cup of coffee (or tea). 
```
docker pull nvcr.io/nvidia/pytorch:19.06-py3
```

If operating on Windows, before testing your docker setup, go into Docker Settings->Shared Drives and click the checkbox next to the drive holding the cloned/downloaded code repository.  A more detailed walkthrough can be found in [this tutorial](https://token2shell.com/howto/docker/sharing-windows-folders-with-containers/).



### Test your environment
Test your docker setup by navigating into the `manifold_learning` directory that you cloned above and executing the following command:
```
sudo ./run_docker.sh
```
This command starts a bash terminal within the Docker environment. You should now be able to execute commands within the docker environment. Make sure that you can view the `manifold_learning` code. 

You can exit the notebook by executing `exit` from the terminal.

### Running jupyter notebooks
Run your jupyter notebook server by navigating into the `manifold_learning` and executing the command below.
```
sh ./run_jupyter.sh
```
This script internally spins up docker and hosts a jupyter notebook server which you can access via your browser. 

Navigate to http://localhost:8888 in your browswer. The token is `manifold`.

You can learn the basics of jupyter notebooks here: https://realpython.com/jupyter-notebook-introduction/. Use `shift+enter` to execute cell.


