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

def prepare_numbers2():
        
    path = "numbers4/gradient_"

    for ii in range(1,10):
        first_file_path = path + str(ii)+"_1.npy"
        second_file_path = path + str(ii)+"_2.npy"
        third_file_path = path + str(ii)+"_3.npy"
        with open(first_file_path, 'rb') as file_1:
            first_array = numpy.load(file_1)
        with open(second_file_path, 'rb') as file_2:
            second_array = numpy.load(file_2)
        with open(third_file_path, 'rb') as file_3:
            third_array = numpy.load(file_3)

        width = len(first_array[0])
        height = len(first_array)

        print(f"width: {width} height: {height}")

        result_array = [[0]*width for ii in range(height)]
        print(f"result_array width: {len(result_array[0])}  height: {len(result_array)}")

        #result_array = int((first_array+second_array+third_array)/3)

        for row in range(height):
            for col in range(width):
                result_array[row][col] = int(round((first_array[row][col] + second_array[row][col] + third_array[row][col])/3))
                #if (first_array[row][col] + second_array[row][col] + third_array[row][col])>260:
                #    result_array[row][col] = 255
                #else:
                #    result_array[row][col] = 0
            
        filepath = path + str(ii) +'.npy'

        with open (filepath, 'wb') as f:
            numpy.save(f, result_array)
        
        result_array = numpy.array(result_array)
        result_im = Image.fromarray(result_array)
        result_im.show()






if __name__=="__main__":
    #prepare_numbers()
    prepare_numbers2()