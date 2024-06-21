## Alternative Text Generator 
- generates alternative text given an image using the blib (bootstrapping language-image pre-training) libary. 
to run: python camption.py 

### Example: 
![home page](images/home_page.JPG)
![Dog image with loading icon](images/generating.JPG)
![Dog image alternative text generated](images/generated.JPG)

### Second example:
![Chicago image with loading icon](images/chi.JPG)
![Chicago image alternative text generated](images/chi_generated.JPG)




### notes:
Need to have the blib library (https://huggingface.co/docs/transformers/en/model_doc/blip) 

package dependencies: 
python -m pip install 
keras
tensorflow
pillow 
scipy
