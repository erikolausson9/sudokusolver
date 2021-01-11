from PIL import Image
from sudoku_solver import sobel_convolution, gaussian_blur
import numpy

def prepare_numbers():
    path = "numbers/"

    for number in range(1,10):
        print(f"Opening image file {number}")
        image_file = Image.open(path+'number_'+ str(number)+'.jpg').convert("L")
        image_array = numpy.asarray(image_file, dtype='int32')
        blurred_array = gaussian_blur(image_array)

      

        cell_mean = numpy.mean(numpy.mean(blurred_array))
        blurred_array[blurred_array>cell_mean*0.77] = 255
        blurred_array[blurred_array<=cell_mean*0.77] = 0

        blurred_im = Image.fromarray(blurred_array)
        blurred_im.show()

        #gradient_array = sobel_convolution(image_array)

        filepath = path + 'number_' + str(number) +'.npy'

        with open (filepath, 'wb') as f:
            numpy.save(f, blurred_array)


    #(width, height) = image_file.size
    #print(f"width: {width}")



if __name__=="__main__":
    prepare_numbers()