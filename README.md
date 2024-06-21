Alternative Text Generator 
- generates alternative text given an image using the blib (bootstrapping language-image pre-training) libary. 
to run: python camption.py 

Example: 
![home page](http://url/to/img.png)
![Dog image with loading icon](http://url/to/img.png)
![Dog image alternative text generated](http://url/to/img.png)

![Chicago image with loading icon](http://url/to/img.png)
![Chicago image alternative text generated](http://url/to/img.png)






notes --------------------------------------------
need to have the blib library (https://huggingface.co/docs/transformers/en/model_doc/blip) 

package dependencies: 
python -m pip install 
keras
tensorflow
pillow 
scipy
