import numpy
from PIL import Image
import math
import sys
import os
import cv2

class CornerPosition:
    
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.sideCount = 0
        self.sideLength = 0

class SudokuImage:

    def __init__(self, image_path):
        self.ratio = 1
        self.image_file = Image.open(image_path).convert("L")
        


    def handle_image(self):
        """
        Downscale input image as preperation to find corners
        """

        print("handling image")

        (width, height) = self.image_file.size
    
        if width<height:
            if width > 300:
                self.ratio = 300/width
        else:
            if height > 300:
                self.ratio = 300/height
 

        resized_image_file = self.image_file.resize((int(width*self.ratio), int(height*self.ratio)))

        print(f"ratio: {self.ratio}")
        image_array = numpy.asarray(resized_image_file, dtype='int32')

        return image_array


    def normalize_image(self, path_to_image, corner_position_list, desired_output_width):
        """
        Normalize size and perspective of actual soduku board
        """

        self.image_file = cv2.imread(path_to_image, 0) #loads in grayscale

        (height, width) = self.image_file.shape
        #print(f"height {height} width {width}")
        
        downscaled_sudoku_size = (corner_position_list[1].col - corner_position_list[0].col)/2 + (corner_position_list[2].row-corner_position_list[1].row)/2
        original_sudoku_size = downscaled_sudoku_size/self.ratio
        original_to_desired_ratio = desired_output_width/original_sudoku_size
        downscaled_to_desired_ratio = original_to_desired_ratio/self.ratio

        resized_image_file = cv2.resize(self.image_file,(int(width*original_to_desired_ratio), int(height*original_to_desired_ratio))) 
        desired_array = numpy.asarray(resized_image_file, dtype='int32')
       
        src_points = numpy.zeros((4,2), dtype = "float32")
        dst_points = numpy.array([[0,0],[desired_output_width,0],[desired_output_width,desired_output_width],[0,desired_output_width]], dtype='float32')

        for ii in range(4):
            corner_position_list[ii].row = int(round(corner_position_list[ii].row * downscaled_to_desired_ratio))
            corner_position_list[ii].col = int(round(corner_position_list[ii].col * downscaled_to_desired_ratio))

            src_points[ii][0]=corner_position_list[ii].col
            src_points[ii][1]=corner_position_list[ii].row

            #Indicate the corner positions
            desired_array[corner_position_list[ii].row-4, corner_position_list[ii].col-4:corner_position_list[ii].col+5]=255
            desired_array[corner_position_list[ii].row+4, corner_position_list[ii].col-4:corner_position_list[ii].col+4]=255
            desired_array[corner_position_list[ii].row-4: corner_position_list[ii].row+4,corner_position_list[ii].col-4]=255
            desired_array[corner_position_list[ii].row-4: corner_position_list[ii].row+4,corner_position_list[ii].col+4]=255
       
        corner_im = Image.fromarray(desired_array)
        corner_im.show()

        M = cv2.getPerspectiveTransform(src_points, dst_points)
        warped = cv2.warpPerspective(resized_image_file, M, (desired_output_width, desired_output_width))

        warped_im = Image.fromarray(warped)
        warped_im.show()

        return warped


def sobel_convolution(input_array):
    

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

    #print("done")
    result_array = numpy.array(result_array)
    result_im = Image.fromarray(result_array)
    #result_im.show()

    #save_numpy_array(result_array, 'lower_left_corner_intermediate_13_13.npy')

    return result_array

def save_numpy_array(array, filepath):

    """
    Auxiliary method used to save numpy arrays
    """

    with open (filepath, 'wb') as f:
        numpy.save(f, array)

def gaussian_blur(input_array):

    width = len(input_array[0])
    height = len(input_array)
    input_array = numpy.array(input_array)
    result_array = numpy.array([[0]*(width-2) for ii in range(height-2)])

    for row in range(1,height-1):
        for col in range(1,width-1):
            matrix_sum = input_array[row-1, col-1] + 2*input_array[row-1, col]+input_array[row-1, col+1]\
                    + 2*input_array[row, col-1] + 4*input_array[row, col] + 2* input_array[row, col+1]\
                    + input_array[row+1, col-1] + 2* input_array[row+1, col] + input_array[row+1, col+1]
            
            result_array[row-1, col-1] = int(round(matrix_sum/16))
                  

    return result_array

def find_corners(input_array):
    """
    Find corners in given input_array, using 13 by 13 gradient images of the four corners.
    """
    print("Finding corners in input image")

    height = len(input_array)
    width = len(input_array[0])
    mid_height = int(height/2)
    mid_width = int(width/2)

    corner_positions = []

    #Search for the corners in a clockwise manner starting from the top left corner

    #Top left corner
    (best_row, best_col) = sub_image_search(input_array, 'top_left_corner_intermediate_13_13.npy', 0, mid_height, 0, mid_width)
    corner_positions.append(CornerPosition(best_row, best_col))

    #Top right corner
    (best_row, best_col) = sub_image_search(input_array, 'top_right_corner_intermediate_13_13.npy', 0, mid_height, mid_width, width)
    corner_positions.append(CornerPosition(best_row, best_col))

    #Lower right corner
    (best_row, best_col) = sub_image_search(input_array, 'lower_right_corner_intermediate_13_13.npy', mid_height, height, mid_width, width)
    corner_positions.append(CornerPosition(best_row, best_col))

    #Lower left corner
    (best_row, best_col) = sub_image_search(input_array, 'lower_left_corner_intermediate_13_13.npy', mid_height, height, 0, mid_width)
    corner_positions.append(CornerPosition(best_row, best_col))
   
    return corner_positions


def sub_image_search(input_array, sub_image_path, start_row, end_row, start_col, end_col):

    """
    Auxiliary method used by find_corners to seek out specified sub_image in input image.
    Parameter sub_image_path should be path to binary file of numpy array.
    Parameters start_row, end_row, start_col and end_col should specify the search area in the input_image. 
    """

    print(f"Finding {sub_image_path} in input image. Start_row: {start_row} end_row: {end_row}, start_col: {start_col}, end_col: {end_col}")
    with open(sub_image_path,'rb') as f:
        sub_image = numpy.load(f)
    

    height = end_row-start_row 
    width  = end_col-start_col
    sub_image_width = len(sub_image[0])
    sub_image_height = len(sub_image)

    best_score = 10000000000 #arbitrarily high number that we are sure to beat
    best_x = 0
    best_y = 0

    diff_array = [[0]*(width-sub_image_width) for ii in range((height-sub_image_height))]
    #step = 2
    for row in range(height-sub_image_height):
        
        if row%10==0:
            print(f"Processing row: {row} of {end_row-start_row-sub_image_height}")
        
        current_row = start_row + row + sub_image_height/2
        for col in range(width-sub_image_width):
            current_col = start_col + col + sub_image_width/2
            difference = 0

            current_matrix = input_array[int(current_row-sub_image_height/2):int(current_row+sub_image_height/2),int(current_col-sub_image_width/2):int(current_col+sub_image_width/2)]
        

            difference_matrix = abs(current_matrix-sub_image)
            difference = sum(sum(difference_matrix))

            if difference < best_score:
                best_score = difference
                best_x = col
                best_y = row
                best_total_col = int(current_col)
                best_total_row = int(current_row) 

            diff_array[row][col] = difference
    
    diff_array = numpy.array(diff_array)
    diff_array = 255-(diff_array/(diff_array.max()/255.0)) #This will normalize the array to [0,255] with highest value given to (x,y) of best match
    
    #Draw a square around the best_x and best_y position to highlight the guess
    if 2 < best_x < (len(diff_array[0])-2) and 2< best_y< (len(diff_array)-2):
        diff_array[best_y-2, best_x-2:best_x+3] = 255
        diff_array[best_y+2, best_x-2:best_x+3] = 255
        diff_array[best_y-2:best_y+3, best_x-2] = 255
        diff_array[best_y-2:best_y+3, best_x+2] = 255
    #diff_im = Image.fromarray(diff_array)
    #diff_im.show() #Will show "heatmap" of best corner match
    print(f"Best x: {best_x} best y: {best_y} best_score: {best_score}")

    #Draw a square around the best_x and best_y position to highlight the guess
    if 2 < best_total_col < (len(input_array[0])-2) and 2< best_total_row< (len(input_array)-2):
        input_array[best_total_row-2, best_total_col-2:best_total_col+3] = 255
        input_array[best_total_row+2, best_total_col-2:best_total_col+3] = 255
        input_array[best_total_row-2:best_total_row+3, best_total_col-2] = 255
        input_array[best_total_row-2:best_total_row+3, best_total_col+2] = 255
    #corner_im = Image.fromarray(input_array)
    #corner_im.show() #Will show "heatmap" of best corner match
    

    return (best_total_row, best_total_col) #return tuple of row and col for best guess

def evaluate_corners(input_array, corner_position_list):

    if (len(corner_position_list)<4):
        print("Error: missing corner in corner_position_list")
        return
    
    margin_of_error = 15 #Number of pixels we can be off (due to perspective change etc) and still assume the corner is correct

    if(abs(corner_position_list[0].row-corner_position_list[1].row)<margin_of_error):
        corner_position_list[0].sideCount += 1
        corner_position_list[1].sideCount += 1
        corner_position_list[0].sideLength = abs(corner_position_list[0].col-corner_position_list[1].col) 
    
    if(abs(corner_position_list[1].col-corner_position_list[2].col)<margin_of_error):
        corner_position_list[1].sideCount += 1
        corner_position_list[2].sideCount += 1
        corner_position_list[1].sideLength = abs(corner_position_list[1].row-corner_position_list[2].row) 

    
    if(abs(corner_position_list[2].row-corner_position_list[3].row)<margin_of_error):
        corner_position_list[2].sideCount += 1
        corner_position_list[3].sideCount += 1
        corner_position_list[2].sideLength = abs(corner_position_list[2].col-corner_position_list[3].col) 


    if(abs(corner_position_list[3].col-corner_position_list[0].col)<margin_of_error):
        corner_position_list[3].sideCount += 1
        corner_position_list[0].sideCount += 1
        corner_position_list[3].sideLength = abs(corner_position_list[3].row-corner_position_list[0].row) 

    corner_level_count = [0]*3 

    for ii in range(4):
        corner_level_count[corner_position_list[ii].sideCount] += 1 
    #corner_level_count[2] holds the number of corners with sideCount == 2, that is with correct 90 degree angles.

    for ii in range(4):
        print(f"corner: {ii}, col: {corner_position_list[ii].col} row: {corner_position_list[ii].row} sideCount: {corner_position_list[ii].sideCount}")  

    if corner_level_count[2]==0:
        print("Unable to locate enough corners in image of sudoku. Bailing out!")
        sys.exit()
    elif corner_level_count[2] == 4:
        print("Hurray, all four corners detected without problems.")
    elif (corner_level_count[2]==2 or corner_level_count[2]==1 ) and corner_level_count[1]==2: #only one corner is incorrectly placed. This we can correct.
        corner_position_list = rectify_corner(input_array, corner_position_list, corner_level_count[2], margin_of_error)
    #elif corner_level_count[2]==1 and corner_level_count[1]==2: #only one corner is incorrectly placed. This we can correct.
    #    corner_position_list = rectify_corner(input_array, corner_position_list, corner_level_count[2], margin_of_error)
    else:
        print("Weird sideCounts. Bailing out!")
        sys.exit()
        
    for ii in range(4):
        print(f"corner: {ii}, col: {corner_position_list[ii].col} row: {corner_position_list[ii].row} sideCount: {corner_position_list[ii].sideCount}")  

    return corner_position_list

def rectify_corner(input_array, corner_position_list, number_of_secure_corners, margin_of_error):
    """
    Auxiliary function used by evaluate_corners to rectify one incorrectly placed corner

    """

    print("Rectifying one corner")

    if number_of_secure_corners == 1:
        #only one secure corner. This means the incorrect corner is always the opposite one

        if corner_position_list[0].sideCount==2:
            guess_row = corner_position_list[3].row
            guess_col = corner_position_list[1].col
            (corner_position_list[2].row, corner_position_list[2].col)=\
                    sub_image_search(input_array, 'lower_right_corner_intermediate_13_13.npy', \
                        max(0, guess_row-margin_of_error), min(len(input_array),guess_row+margin_of_error), max(0, guess_col-margin_of_error), min(len(input_array[0]), guess_col+margin_of_error))
        elif corner_position_list[1].sideCount==2:
            guess_row = corner_position_list[2].row
            guess_col = corner_position_list[0].col
            (corner_position_list[3].row, corner_position_list[3].col) = \
                    sub_image_search(input_array, 'lower_left_corner_intermediate_13_13.npy', \
                        max(0, guess_row-margin_of_error), min(len(input_array),guess_row+margin_of_error), max(0, guess_col-margin_of_error), min(len(input_array[0]), guess_col+margin_of_error))
        elif corner_position_list[2].sideCount==2:
            guess_row = corner_position_list[1].row
            guess_col = corner_position_list[3].col
            (corner_position_list[0].row, corner_position_list[0].col) = \
                    sub_image_search(input_array, 'top_left_corner_intermediate_13_13.npy', \
                        max(0, guess_row-margin_of_error), min(len(input_array),guess_row+margin_of_error), max(0, guess_col-margin_of_error), min(len(input_array[0]), guess_col+margin_of_error))
        else:
            guess_row = corner_position_list[0].row
            guess_col = corner_position_list[2].col
            (corner_position_list[1].row, corner_position_list[1].col)=\
                    sub_image_search(input_array, 'top_right_corner_intermediate_13_13.npy', \
                        max(0, guess_row-margin_of_error), min(len(input_array),guess_row+margin_of_error), max(0, guess_col-margin_of_error), min(len(input_array[0]), guess_col+margin_of_error))

    
    else:
        #two secure corners, this means we have to look at distances as well to decide which corner is incorrect

        if corner_position_list[0].sideCount ==2 and corner_position_list[1].sideCount==2:
            secure_side_length = corner_position_list[0].sideLength
            if abs(secure_side_length-corner_position_list[3].sideLength)<abs(secure_side_length-corner_position_list[1].sideLength):
                guess_row = corner_position_list[3].row
                guess_col = corner_position_list[1].col
                (corner_position_list[2].row, corner_position_list[2].col)=\
                    sub_image_search(input_array, 'lower_right_corner_intermediate_13_13.npy', \
                        max(0, guess_row-margin_of_error), min(len(input_array),guess_row+margin_of_error), max(0, guess_col-margin_of_error), min(len(input_array[0]), guess_col+margin_of_error))
            else:
                guess_row = corner_position_list[2].row
                guess_col = corner_position_list[1].col
                (corner_position_list[3].row, corner_position_list[3].col) = \
                    sub_image_search(input_array, 'lower_left_corner_intermediate_13_13.npy', \
                        max(0, guess_row-margin_of_error), min(len(input_array),guess_row+margin_of_error), max(0, guess_col-margin_of_error), min(len(input_array[0]), guess_col+margin_of_error))

        elif corner_position_list[1].sideCount ==2 and corner_position_list[2].sideCount==2:
            secure_side_length = corner_position_list[1].sideLength
            if abs(secure_side_length-corner_position_list[0].sideLength)<abs(secure_side_length-corner_position_list[2].sideLength):
                guess_row = corner_position_list[2].row
                guess_col = corner_position_list[0].col
                (corner_position_list[3].row, corner_position_list[3].col)=\
                    sub_image_search(input_array, 'lower_left_corner_intermediate_13_13.npy', \
                        max(0, guess_row-margin_of_error), min(len(input_array),guess_row+margin_of_error), max(0, guess_col-margin_of_error), min(len(input_array[0]), guess_col+margin_of_error))
            else:
                guess_row = corner_position_list[1].row
                guess_col = corner_position_list[3].col
                (corner_position_list[0].row, corner_position_list[0].col) = \
                    sub_image_search(input_array, 'top_left_corner_intermediate_13_13.npy', \
                        max(0, guess_row-margin_of_error), min(len(input_array),guess_row+margin_of_error), max(0, guess_col-margin_of_error), min(len(input_array[0]), guess_col+margin_of_error))

        elif corner_position_list[2].sideCount ==2 and corner_position_list[3].sideCount==2:
            secure_side_length = corner_position_list[2].sideLength
            if abs(secure_side_length-corner_position_list[2].sideLength)<abs(secure_side_length-corner_position_list[1].sideLength):
                guess_row = corner_position_list[0].row
                guess_col = corner_position_list[2].col
                (corner_position_list[1].row, corner_position_list[1].col)=\
                    sub_image_search(input_array, 'top_right_corner_intermediate_13_13.npy', \
                        max(0, guess_row-margin_of_error), min(len(input_array),guess_row+margin_of_error), max(0, guess_col-margin_of_error), min(len(input_array[0]), guess_col+margin_of_error))
            else:
                guess_row = corner_position_list[1].row
                guess_col = corner_position_list[3].col
                (corner_position_list[0].row, corner_position_list[0].col) = \
                    sub_image_search(input_array, 'top_left_corner_intermediate_13_13.npy', \
                        max(0, guess_row-margin_of_error), min(len(input_array),guess_row+margin_of_error), max(0, guess_col-margin_of_error), min(len(input_array[0]), guess_col+margin_of_error))

        else: #secure corners have index 3 and 0
            
            secure_side_length = corner_position_list[3].sideLength
            if abs(secure_side_length-corner_position_list[0].sideLength)<abs(secure_side_length-corner_position_list[2].sideLength):
                guess_row = corner_position_list[3].row
                guess_col = corner_position_list[1].col
                (corner_position_list[2].row, corner_position_list[2].col)=\
                    sub_image_search(input_array, 'lower_right_corner_intermediate_13_13.npy', \
                        max(0, guess_row-margin_of_error), min(len(input_array),guess_row+margin_of_error), max(0, guess_col-margin_of_error), min(len(input_array[0]), guess_col+margin_of_error))
            else:
                guess_row = corner_position_list[0].row
                guess_col = corner_position_list[2].col
                (corner_position_list[1].row, corner_position_list[1].col) = \
                    sub_image_search(input_array, 'top_right_corner_intermediate_13_13.npy', \
                        max(0, guess_row-margin_of_error), min(len(input_array),guess_row+margin_of_error), max(0, guess_col-margin_of_error), min(len(input_array[0]), guess_col+margin_of_error))

    for ii in range(4):
        #update the sideLengths given the new corner_positions
        if ii%2==0:
            corner_position_list[ii].sideLength = abs(corner_position_list[ii+1].col-corner_position_list[ii].col)
        else:
            corner_position_list[ii].sideLength = abs(corner_position_list[(ii+1)%4].row-corner_position_list[ii].row)

    return corner_position_list


def segment_cells(normalized_array):
    
    width = len(normalized_array)
    cell_width = int(round(width/9))
    cell_margin = int(round(cell_width*0.12))

    #load list of thresholded images of the numbers to compare to:
    #true_numbers = []
    #for ii in range(1,10):
    #    path =  'numbers2/number_' + str(ii) + '.npy'
    #    with open(path,'rb') as f:
    #        true_numbers.append(numpy.load(f))

    true_gradient_numbers = []
    for ii in range(1,10):
        path =  'numbers/gradient_' + str(ii) + '.npy'
        with open(path,'rb') as f:
            true_gradient_numbers.append(numpy.load(f))

    #Create a solution-matrix to store the found numbers
    solution_matrix = [[0]*9 for ii in range(9)]

    for cell_row in range(0,9):
        for cell_col in range(0,9):
    
            current_cell = normalized_array[cell_row*cell_width+cell_margin:(cell_row+1)*cell_width-cell_margin, cell_col*cell_width+cell_margin:(cell_col+1)*cell_width-cell_margin]
            cell_mean = numpy.mean(numpy.mean(current_cell))
            
            current_cell = gaussian_blur(current_cell)
            current_cell_for_gradient = current_cell.copy()

            #Threshold the cell and take a sample at its center
            current_cell[current_cell>cell_mean*0.81] = 255
            current_cell[current_cell<=cell_mean*0.81]=0
            cell_sample_margin = int(round(cell_width*0.35))
            cell_sample = current_cell[cell_sample_margin:cell_width-cell_sample_margin, cell_sample_margin:cell_width-cell_sample_margin]
            
            if numpy.mean(numpy.mean(cell_sample))<250: #This is true if we have a number in the current cell
                #find the bounding box of the thresholded number
                
                #Top row
                start_row = 0
                current_row = current_cell[0,:]
                number_of_black_pixels = len(numpy.where(current_row<250)[0])
                while number_of_black_pixels < 2:
                    start_row += 1
                    current_row = current_cell[start_row, :]
                    number_of_black_pixels = len(numpy.where(current_row<250)[0])
    
                #Leftmost column
                start_col = 0
                current_col = current_cell[:,0]
                number_of_black_pixels = len(numpy.where(current_col<250)[0])
                while number_of_black_pixels < 2:
                    start_col += 1
                    current_col = current_cell[:, start_col]
                    number_of_black_pixels = len(numpy.where(current_col<250)[0])

                #Bottom row
                end_row = len(current_cell)
                current_row = current_cell[-1,:]
                number_of_black_pixels = len(numpy.where(current_row<250)[0])
                while number_of_black_pixels < 2:
                    end_row -= 1
                    current_row = current_cell[end_row, :]
                    number_of_black_pixels = len(numpy.where(current_row<250)[0])
                
                #Rightmost column
                end_col = len(current_cell[0])
                current_col = current_cell[:, -1]
                number_of_black_pixels = len(numpy.where(current_col<250)[0])
                while number_of_black_pixels < 2:
                    end_col -= 1
                    current_col = current_cell[:, end_col]
                    number_of_black_pixels = len(numpy.where(current_col<250)[0])
                
                #cut out the number and resize to 16 pixels width and 22 pixels height
                #normalized_cell = current_cell[start_row:end_row+1,start_col:end_col+1]
                #normalized_cell = Image.fromarray(normalized_cell)
                #normalized_cell.show()
                #normalized_cell = normalized_cell.resize((16,22))
               

                #cut out for gradient number image
                start_row = max(0, (start_row-2))
                start_col = max(0, (start_col-2))
                end_row = min(len(current_cell), (end_row+3))
                end_col = min(len(current_cell), (end_col+3))
                normalized_gradient_cell = current_cell_for_gradient[start_row:end_row, start_col:end_col]
                normalized_gradient_cell = sobel_convolution(normalized_gradient_cell)
           
                normalized_gradient_cell = normalized_gradient_cell/(normalized_gradient_cell.max()/255.0) #This will normalize the array to [0,255] with highest value given to (x,y) of best match
                
                #Normalize size of number image
                gradient_cell_im = Image.fromarray(normalized_gradient_cell)
                gradient_cell_im = gradient_cell_im.resize((18,24))
                normalized_gradient_cell = numpy.asarray(gradient_cell_im)
              
                solution_matrix[cell_row][cell_col] = identify_number(normalized_gradient_cell, true_gradient_numbers)
                
                #Use save_number to save image and numpy array for training purposes etc. 
                #save_number(normalized_gradient_cell, 'numbers5', cell_row, cell_col, guess)

    return solution_matrix          
                


def save_number(number_array, directory, row, col, number=0):
    if number == 0:
        path = f"{directory}/row_{row}_col_{col}"
    else:
        root_path = f"{directory}/{number}"
        count = 0
        path = root_path
        while (os.path.exists(path + '.jpg') or os.path.exists(path + '.npy')):
            count += 1
            path = f"{root_path}_{count}"
    
    save_numpy_array(number_array, path + '.npy')
    number_image = Image.fromarray(number_array)
    if number_image.mode != 'RGB':
        number_image = number_image.convert('RGB')
    
    number_image.save(path + '.jpg')


def identify_number(single_cell_input_array, true_numbers):
    
    height = len(single_cell_input_array)
    width = len(single_cell_input_array[0])

    sub_image_height = len(true_numbers[0])
    sub_image_width = len(true_numbers[0][0])

    if height != sub_image_height or width != sub_image_width:
        print(f"Error: image and sub_image dimensions don't match. Height: {height} sub_image_height: {sub_image_height} Width: {width} sub_image_width: {sub_image_width}")
    

    best_score = 100000000
    second_best_score = best_score
    guess = 0
    second_guess = 0

    for number in range(len(true_numbers)):
        
        difference_matrix = (single_cell_input_array-true_numbers[number])*(single_cell_input_array-true_numbers[number])
        difference = sum(sum(difference_matrix))

        if difference < best_score:
            second_best_score = best_score
            second_guess = guess
            best_score = difference
            guess = number+1
        
        elif difference < second_best_score:
            second_best_score = difference
            second_guess = number+1
        
    
    print(f"Guess: {guess} and second guess: {second_guess} diff percentage: {round(second_best_score/best_score,2)}")

    return guess


def print_solution_matrix(solution_matrix):
    print("Solution matrix: ")
    print('-------------------')
    for row in range(9):
        #print('|', end='')
        #for col in range(9):
        #    if solution_matrix[row][col]>0:
        #        print(solution_matrix[row][col], end='')
        #    else:
        #        print(' ', end='')
        #    if col%3==2:
        #        print('|', end='')

        #print(' ')
        print(f"|{solution_matrix[row][0]} {solution_matrix[row][1]} {solution_matrix[row][2]}|{solution_matrix[row][3]} {solution_matrix[row][4]} {solution_matrix[row][5]}|{solution_matrix[row][6]} {solution_matrix[row][7]} {solution_matrix[row][8]}|")
        if row%3==2:
            print('-------------------')

def print_working_matrix(working_matrix):

    print("Working matrix:")
    print('-----------------')
    for row in range(9):
        max_length = 0
        for col in range(9):
            if len(working_matrix[row][col])>max_length:
                max_length = len(working_matrix[row][col])

        for ii in range(max_length):
            print('|', end='')
            for col in range(9):
                
                if ii<len(working_matrix[row][col]):
                    print(working_matrix[row][col][ii], end='')
                else:
                    print(' ', end='')
                if col%3==2:
                    print('|', end='')
            print()

            
        print('--------------------')


def solve_sudoku(solution_matrix):

    working_matrix = [[0]*9 for ii in range(9)]
    for row in range(9):
        for col in range(9):
            if solution_matrix[row][col]>0:
                working_matrix[row][col] = [solution_matrix[row][col]]
            else:
                working_matrix[row][col] = [1,2,3,4,5,6,7,8,9]
    
    print_working_matrix(working_matrix)
    pass_count = 1

    while 0 in solution_matrix and pass_count <2:
        for row in range(9):
            for col in range(9):
                for ii in range(len(working_matrix[row][col])):
                    current_number = working_matrix[row][col][ii]
                    if current_number in solution_matrix[row][:] or current_number in solution_matrix[:][col]:
                        working_matrix[row][col].remove(current_number)
        pass_count += 1
    
    print_working_matrix(working_matrix)
