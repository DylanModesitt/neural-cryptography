# neural-cryptography

These are tools experimenting with neural cryptography. 
This is being done as the final project to an MIT course, 6.857, 
and is still in active development. Once completed, this 
ReadMe will include a comprehensive summary of our
findings and results. 

# Setup

After cloning this repository, setup a python virtual environment. This can be done with

```
source setup
```

this can also be used after initially setting up the environment to activate the virtual environment.

# Project Structure

The project is structured in the following fashion.

```
.
+-- _setup
+-- _bin
|   +-- this includes binary files like weights,
|      cached data files, or visualizations
+-- _data
|   +-- data.py
|        +-- data generation programs 
+-- _genneral
|        +-- general software like custom keras layers,
|            utility functions, etc.
+-- _models
|   +-- code that builds, trains, and visualizes keras

```
