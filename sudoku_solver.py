import numpy
from PIL import Image
import math

class CornerPosition:
    
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.sideCount = 0



def handle_image(image_path):
    print("handling image")
    image_file = Image.open(image_path).convert("L")

    (width, height) = image_file.size
    print(f"width: {width}")
    
    ratio = 1
    if width<height:
        if width > 300:
            ratio = 300/width
    else:
        if height > 300:
            ratio = 300/height
 

    image_file = image_file.resize((int(width*ratio), int(height*ratio)))

    print(image_file.format)
    print(image_file.size)
    print(image_file.mode)
    #image_file.show()

    image_array = numpy.asarray(image_file, dtype='int32')

    print(image_array)

    print(type(image_array))

    #gr_image = Image.fromarray(image_array)

    #gr_image.show()
    return image_array

def sobel_convolution(input_array):
    print(f"Starting sobel convultion on {len(input_array)} x {len(input_array[0])} array ")

    width = len(input_array[0])
    height = len(input_array)

    result_array = [[0]*(width-2) for ii in range(height-2)]

    for row in range(height):
        for col in range(width):

            if(row>0 and row<(height-1) and col>0 and col<(width-1)):
                x_convol =  input_array[row-1][col-1] - input_array[row-1][col+1] + (2*input_array[row][col-1]) - (2*input_array[row][col+1]) + input_array[row+1][col-1] - input_array[row+1][col+1]
                #print(type(x_convol))
                y_convol =  input_array[row-1][col-1]-input_array[row+1][col-1]+\
                            2*input_array[row-1][col] - 2*input_array[row+1][col]+\
                            input_array[row-1][col+1] - input_array[row+1][col+1]
                #print(type(y_convol))
                #print(f"row: {row} col: {col} input_array: {input_array[row-1][col-1]},{input_array[row-1][col]},{input_array[row-1][col+1]},{input_array[row][col-1]},{input_array[row][col]},{input_array[row][col+1]},{input_array[row+1][col-1]},{input_array[row+1][col]},{input_array[row+1][col+1]} x_convol: {x_convol} y_convol: {y_convol}")

                result_array[row-1][col-1] = math.sqrt(x_convol*x_convol + y_convol*y_convol)   

            #print(f" row: {row} and column: {column}")

    print("done")
    result_array = numpy.array(result_array)
    result_im = Image.fromarray(result_array)
    result_im.show()

    #save_numpy_array(result_array, 'lower_left_corner_intermediate_13_13.npy')

    return result_array

def save_numpy_array(array, filepath):
    with open (filepath, 'wb') as f:
        numpy.save(f, array)

def find_corners(input_array):
    print("Finding corners in input image")

    height = len(input_array)
    width = len(input_array[0])
    mid_height = int(height/2)
    mid_width = int(width/2)

    #Top left corner
    sub_image_search(input_array, 'top_left_corner_intermediate_13_13.npy', 0, mid_height, 0, mid_width)

    #Top right corner
    sub_image_search(input_array, 'top_right_corner_intermediate_13_13.npy', 0, mid_height, mid_width, width)

    
    #Lower left corner
    sub_image_search(input_array, 'lower_left_corner_intermediate_13_13.npy', mid_height, height, 0, mid_width)
    
    #Lower right corner
    sub_image_search(input_array, 'lower_right_corner_intermediate_13_13.npy', mid_height, height, mid_width, width)


    """
    with open('top_right_corner_downscaled.npy','rb') as f:
        lower_left_corner = numpy.load(f) #should give numpy array of convolution result of image with lower left corner

    #print(lower_left_corner)
    result_im = Image.fromarray(lower_left_corner)
    result_im.show(title="Loaded corner image")

    width = len(input_array[0])
    height = len(input_array)
    corner_image_width = len(lower_left_corner[0])
    corner_image_height = len(lower_left_corner)

    best_score = 10000000000
    best_x = 0
    best_y = 0

    diff_array = [[0]*(width-corner_image_width) for ii in range((height-corner_image_height))]

    for row in range(height-corner_image_height):
        print(f"Processing row: {row} of {height-corner_image_height}") 
        for col in range(width-corner_image_width):
            current_row = row + corner_image_height/2
            current_col = col + corner_image_width/2
            difference = 0
            for sub_row in range(corner_image_height):
                for sub_col in range(corner_image_width):
                    difference = difference + abs(input_array[int(current_row-corner_image_height/2+sub_row)][int(current_col-corner_image_width/2+sub_col)]-lower_left_corner[sub_row][sub_col])

            diff_array[int(row)][int(col)] = difference        
            if difference<best_score:
                best_score = difference
                best_x = current_row
                best_y = current_col
    
    print(f"Done with sub_image search. Best x: {best_x} best y: {best_y} best_score: {best_score}")
    
    diff_array = numpy.array(diff_array)
    diff_array = 255-(diff_array/(diff_array.max()/255.0)) #This will normalize the array to [0,255] with highest value given to (x,y) of best match
    diff_im = Image.fromarray(diff_array)
    diff_im.show() #Will show "heatmap" of best corner match
    
    """

def sub_image_search(input_array, sub_image_path, start_row, end_row, start_col, end_col):

    """
    Auxiliary method used by find_corners to seek out specified sub_image in input image.
    Parameter sub_image_path should be path to binary file of numpy array.
    Parameters start_row, end_row, start_col and end_col should specify the search area in the input_image. 
    """

    print(f"Finding {sub_image_path} in input image. Start_row: {start_row} end_row: {end_row}, start_col: {start_col}, end_col: {end_col}")
    with open(sub_image_path,'rb') as f:
        sub_image = numpy.load(f)
    
    result_im = Image.fromarray(sub_image)
    result_im.show(title="Loaded sub_image")

    height = end_row-start_row 
    width  = end_col-start_col
    sub_image_width = len(sub_image[0])
    sub_image_height = len(sub_image)

    best_score = 10000000000
    best_x = 0
    best_y = 0

    diff_array = [[0]*(width-sub_image_width) for ii in range((height-sub_image_height))]

    for row in range(height-sub_image_height):
        print(f"Processing row: {row} of {end_row-start_row-sub_image_height}")
        current_row = start_row + row + sub_image_height/2
        for col in range(width-sub_image_width):
            current_col = start_col + col + sub_image_width/2
            difference = 0

            current_matrix = input_array[int(current_row-sub_image_height/2):int(current_row+sub_image_height/2),int(current_col-sub_image_width/2):int(current_col+sub_image_width/2)]
            
            #print(int(current_row-sub_image_height/2))
            #print(int(current_row+sub_image_height/2))
            #print(int(current_col-sub_image_width/2))
            #print(int(current_col+sub_image_width/2))
            #print(current_matrix.size)

            difference_matrix = abs(current_matrix-sub_image)
            difference = sum(sum(difference_matrix))

            if difference < best_score:
                best_score = difference
                best_x = col
                best_y = row

            diff_array[row][col] = difference
    
    diff_array = numpy.array(diff_array)
    diff_array = 255-(diff_array/(diff_array.max()/255.0)) #This will normalize the array to [0,255] with highest value given to (x,y) of best match
    
    print(len(diff_array))
    print(len(diff_array[0]))

    if 2 < best_x < (len(diff_array[0])-2) and 2< best_y< (len(diff_array)-2):
        diff_array[best_y-2, best_x-2:best_x+3] = 255
        diff_array[best_y+2, best_x-2:best_x+3] = 255
        diff_array[best_y-2:best_y+3, best_x-2] = 255
        diff_array[best_y-2:best_y+3, best_x+2] = 255
    diff_im = Image.fromarray(diff_array)
    diff_im.show() #Will show "heatmap" of best corner match
    print(f"Best x: {best_x} best y: {best_y} best_score: {best_score}")