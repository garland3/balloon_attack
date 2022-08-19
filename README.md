# Balloon Attack

* fork of https://github.com/wmcnally/kapao
    - very nice work
    - all the code ran on the 1st try

* setup using the setup section below
* changed the `LoadImages` ro `LoadWebCam`
* added output folder as a arg. 
`python demos/image.py --pose --face --no-kp-dets --imgsz 256`
* game test
` python demos/game_test.py`

### Setup
1. If you haven't already, [install Anaconda or Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).
2. Create a new conda environment with Python 3.6: `$ conda create -n kapao python=3.6`.
3. Activate the environment: `$ conda activate kapao`
4. Clone this repo: `$ git clone git@github.com:garland3/balloon_attack.git`
5. Install the dependencies: `$ cd balloon_attack && pip install -r requirements2.txt`
6. Download the trained models: `$ python data/scripts/download_models.py`


## Game 2

* starting from real python's tutorial on games. 
* https://github.com/realpython/materials/blob/master/pygame-a-primer/py_tut_with_images.py
* You need a webcam
* pop the ballons with your hands before they get to the top
  *  single points for using hands
  *  double points for using feet
* run with 
```bash
python demos/game2.py
```    

* export requirements
```
pip list --format=freeze > requirements2.txt
```    

## Original Readme
`orginal_readme.md`