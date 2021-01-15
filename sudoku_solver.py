import numpy
from PIL import Image
import math
import sys
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
        print("handling image")
        #image_file = Image.open(image_path).convert("L")

        (width, height) = self.image_file.size
        print(f"width: {width}")
    
    
        if width<height:
            if width > 300:
                self.ratio = 300/width
        else:
            if height > 300:
                self.ratio = 300/height
 

        resized_image_file = self.image_file.resize((int(width*self.ratio), int(height*self.ratio)))

        print(resized_image_file.format)
        print(resized_image_file.size)
        print(resized_image_file.mode)
        #image_file.show()
        print(f"ratio: {self.ratio}")
        image_array = numpy.asarray(resized_image_file, dtype='int32')

    
        return image_array

    def normalize_image(self, path_to_image, corner_position_list, desired_output_width):
        
        self.image_file = cv2.imread(path_to_image, 0) #loads in grayscale

        (height, width) = self.image_file.shape
        print(f"height {height} width {width}")

        #original_array = numpy.asarray(self.image_file, dtype='int32')

        downscaled_sudoku_size = (corner_position_list[1].col - corner_position_list[0].col)/2 + (corner_position_list[2].row-corner_position_list[1].row)/2
        original_sudoku_size = downscaled_sudoku_size/self.ratio
        original_to_desired_ratio = desired_output_width/original_sudoku_size
        downscaled_to_desired_ratio = original_to_desired_ratio/self.ratio

        resized_image_file = cv2.resize(self.image_file,(int(width*original_to_desired_ratio), int(height*original_to_desired_ratio))) 
        desired_array = numpy.asarray(resized_image_file, dtype='int32')
       
        #print(f"downscaled_sudoku_size: {downscaled_sudoku_size} original_sudoku_size: {original_sudoku_size}")

        src_points = numpy.zeros((4,2), dtype = "float32")
        dst_points = numpy.array([[0,0],[desired_output_width,0],[desired_output_width,desired_output_width],[0,desired_output_width]], dtype='float32')

        for ii in range(4):
            corner_position_list[ii].row = int(round(corner_position_list[ii].row * downscaled_to_desired_ratio))
            corner_position_list[ii].col = int(round(corner_position_list[ii].col * downscaled_to_desired_ratio))

            src_points[ii][0]=corner_position_list[ii].col
            src_points[ii][1]=corner_position_list[ii].row

            desired_array[corner_position_list[ii].row-4, corner_position_list[ii].col-4:corner_position_list[ii].col+5]=255
            desired_array[corner_position_list[ii].row+4, corner_position_list[ii].col-4:corner_position_list[ii].col+4]=255
            desired_array[corner_position_list[ii].row-4: corner_position_list[ii].row+4,corner_position_list[ii].col-4]=255
            desired_array[corner_position_list[ii].row-4: corner_position_list[ii].row+4,corner_position_list[ii].col+4]=255
       
        corner_im = Image.fromarray(desired_array)
        corner_im.show()

        M = cv2.getPerspectiveTransform(src_points, dst_points)
        warped = cv2.warpPerspective(resized_image_file, M, (desired_output_width, desired_output_width) )
        print(f"warped size: {len(warped)} and: {len(warped[0])}")



        warped_im = Image.fromarray(warped)
        warped_im.show()
        #warped_im.save("warped_sudoku_extrabild.jpg")

        return warped


def sobel_convolution(input_array):
    #print(f"Starting sobel convultion on {len(input_array)} x {len(input_array[0])} array ")

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
    #print(f"Starting blur convultion on {len(input_array)} x {len(input_array[0])} array ")

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
    
    result_im = Image.fromarray(sub_image)
    #result_im.show(title="Loaded sub_image")

    height = end_row-start_row 
    width  = end_col-start_col
    sub_image_width = len(sub_image[0])
    sub_image_height = len(sub_image)

    best_score = 10000000000
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
                best_total_col = int(current_col) #best_x + start_col
                best_total_row = int(current_row) #best_y + start_row

            diff_array[row][col] = difference
    
    diff_array = numpy.array(diff_array)
    diff_array = 255-(diff_array/(diff_array.max()/255.0)) #This will normalize the array to [0,255] with highest value given to (x,y) of best match
    
    #Draw a square around the best_x and best_y position to highlight the guess
    if 2 < best_x < (len(diff_array[0])-2) and 2< best_y< (len(diff_array)-2):
        diff_array[best_y-2, best_x-2:best_x+3] = 255
        diff_array[best_y+2, best_x-2:best_x+3] = 255
        diff_array[best_y-2:best_y+3, best_x-2] = 255
        diff_array[best_y-2:best_y+3, best_x+2] = 255
    diff_im = Image.fromarray(diff_array)
    #diff_im.show() #Will show "heatmap" of best corner match
    print(f"Best x: {best_x} best y: {best_y} best_score: {best_score}")

    #Draw a square around the best_x and best_y position to highlight the guess
    if 2 < best_total_col < (len(input_array[0])-2) and 2< best_total_row< (len(input_array)-2):
        input_array[best_total_row-2, best_total_col-2:best_total_col+3] = 255
        input_array[best_total_row+2, best_total_col-2:best_total_col+3] = 255
        input_array[best_total_row-2:best_total_row+3, best_total_col-2] = 255
        input_array[best_total_row-2:best_total_row+3, best_total_col+2] = 255
    corner_im = Image.fromarray(input_array)
    #corner_im.show() #Will show "heatmap" of best corner match
    #print(f"Best x: {best_x} best y: {best_y} best_score: {best_score}")


    return (best_total_row, best_total_col) #return tuple of row and col for best guess

def evaluate_corners(corner_position_list):

    if (len(corner_position_list)<4):
        print("Error: missing corner in corner_position_list")
        return
    
    margin_of_error = 15

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

    for ii in range(4):
        print(f"corner: {ii}, col: {corner_position_list[ii].col} row: {corner_position_list[ii].row} sideCount: {corner_position_list[ii].sideCount}")  

    if corner_level_count[2]==0:
        print("Unable to locate enough corners in image of sudoku. Bailing out!")
        sys.exit()
    elif corner_level_count[2] == 4:
        print("Hurray, all four corners detected without problems.")
    elif corner_level_count[2]==2 and corner_level_count[1]==2: #only one corner is incorrectly placed. This we can correct.
        corner_position_list = rectify_corner(corner_position_list, corner_level_count[2])
    elif corner_level_count[2]==1 and corner_level_count[1]==2: #only one corner is incorrectly placed. This we can correct.
        corner_position_list = rectify_corner(corner_position_list, corner_level_count[2])
    else:
        print("Weird sideCounts. Bailing out!")
        sys.exit()
        
    for ii in range(4):
        print(f"corner: {ii}, col: {corner_position_list[ii].col} row: {corner_position_list[ii].row} sideCount: {corner_position_list[ii].sideCount}")  

    return corner_position_list

def rectify_corner(corner_position_list, number_of_secure_corners):
    """
    Auxiliary function used by evaluate_corners to rectify one incorrectly placed corner

    """

    print("Rectifying one corner")

    if number_of_secure_corners == 1:
        #only one secure corner. This means the incorrect corner is always the opposite one

        for ii in range(4):
            if corner_position_list[ii].sideCount == 2:
                if ii%2==0:
                    corner_position_list[(ii+2)%4].row = corner_position_list[(ii+3)%4].row
                    corner_position_list[(ii+2)%4].col = corner_position_list[(ii+1)%4].col
                else:
                    corner_position_list[(ii+2)%4].col = corner_position_list[(ii+3)%4].col
                    corner_position_list[(ii+2)%4].row = corner_position_list[(ii+1)%4].row

    else:
        #two secure corners, this means we have to look at distances as well to decide which corner is incorrect

        for ii in range(4):
            next_index = (ii+1)%4

            if corner_position_list[ii].sideCount==2 and corner_position_list[next_index].sideCount==2:
                secure_side_length = corner_position_list[ii].sideLength
                if abs(secure_side_length-corner_position_list[next_index].sideLength)<abs(secure_side_length-corner_position_list[(ii+3)%4].sideLength):
                    if ii%2==0:
                        corner_position_list[(ii+3)%4].col = corner_position_list[ii].col
                        corner_position_list[(ii+3)%4].row = corner_position_list[(ii+2)%4].row
                    else:
                        corner_position_list[(ii+3)%4].row = corner_position_list[ii].row
                        corner_position_list[(ii+3)%4].col = corner_position_list[(ii+2)%4].col
                else:
                    if ii%2==0:
                        corner_position_list[(ii+2)%4].col = corner_position_list[(ii+1)%4].col
                        corner_position_list[(ii+2)%4].row = corner_position_list[(ii+3)%4].row
                    else:
                        corner_position_list[(ii+2)%4].row = corner_position_list[(ii+1)%4].row
                        corner_position_list[(ii+2)%4].col = corner_position_list[(ii+3)%4].col

    for ii in range(4):
        #update the sideLengths given the new corner_positions
        if ii%2==0:
            corner_position_list[ii].sideLength = abs(corner_position_list[ii+1].col-corner_position_list[ii].col)
        else:
            corner_position_list[ii].sideLength = abs(corner_position_list[(ii+1)%4].row-corner_position_list[ii].row)

    return corner_position_list




def segment_cells(input_array, gradient_array, corner_position_list):

    #load list of gradient images of the numbers to compare to:
    true_numbers = []
    for ii in range(1,10):
        path =  'numbers/number_' + str(ii) + '.npy'
        with open(path,'rb') as f:
            true_numbers.append(numpy.load(f))

    top_length = corner_position_list[0].sideLength
    right_length = corner_position_list[1].sideLength
    bottom_length = corner_position_list[2].sideLength
    left_length = corner_position_list[3].sideLength

    left_start_row = corner_position_list[0].row
    right_start_row = corner_position_list[1].row
    top_start_col = corner_position_list[0].col
    bottom_start_col = corner_position_list[3].col

    

    print("Segmenting cells")
    print(f"top_length: {top_length} right_length: {right_length} bottom_length: {bottom_length} left_length: {left_length}")

    start_on_row = []
    rounding_error_in_column = [0.0]*9
    rounding_error_in_row = [0.0]*9
    for ii in range(9):
        start_on_row.append(int(round((8-ii)*left_start_row/8 + ii*right_start_row/8)))

    for cell_row in range(9):
        row_length = (8-cell_row)*top_length/8 + cell_row*bottom_length/8
        cell_width = int(round(row_length/9.0))
        cell_margin = int(round(cell_width*0.25)) #number of pixels to omit on every side to only get the center of the cell
        start_on_col = int(round((8-cell_row)*top_start_col/8 + cell_row*bottom_start_col/8)) 
        for cell_col in range(9):
            col_length = (8-cell_col)*left_length/8 + cell_col*right_length/8
            cell_height = int(round(col_length/9.0))

            rounding_error_in_row[cell_row] += row_length/9.0-cell_width
            if rounding_error_in_row[cell_row] > 1.0:
                start_on_col += 1
                rounding_error_in_row[cell_row] -= 1.0
            
            rounding_error_in_column[cell_col] += col_length/9.0-cell_height
            if rounding_error_in_column[cell_col] > 1.0:
                start_on_row[cell_col] += 1
                rounding_error_in_column[cell_col] -= 1.0

            #start_on_row = int(round((8-cell_col)*left_start_row/8 + cell_col*right_start_row/8))



            
            #single_cell_array = result_array[start_row + cell_row*cell_width + cell_margin : start_row + (cell_row+1)*cell_width - cell_margin , start_col + cell_col*cell_width + cell_margin : start_col + (cell_col+1)*cell_width - cell_margin]
            #display the bound-boxes on the original array
            #result_array[int(round(start_row + cell_row*cell_width + cell_margin)):int(round(start_row + (cell_row+1)*cell_width - cell_margin)), int(round(start_col + cell_col*cell_height+cell_margin))] = 255
            #result_array[int(round(start_row + cell_row*cell_width + cell_margin)):int(round(start_row + (cell_row+1)*cell_width - cell_margin)), int(round(start_col + (cell_col+1)*cell_height-cell_margin))] = 255
            #result_array[int(round(start_row + cell_row*cell_width + cell_margin)), int(round(start_col + cell_col*cell_height+cell_margin)):int(round(start_col + (cell_col+1)*cell_height-cell_margin))] = 255
            #result_array[int(round(start_row + (cell_row+1)*cell_width - cell_margin)), int(round(start_col + cell_col*cell_height+cell_margin)):int(round(start_col + (cell_col+1)*cell_height-cell_margin))] = 255


            #single_cell_im = Image.fromarray(single_cell_array)
            #single_cell_im.show()

            single_cell_gradient_array = gradient_array[start_on_row[cell_col] + cell_margin:start_on_row[cell_col]+cell_height-cell_margin,start_on_col+cell_margin:start_on_col+cell_width-cell_margin]
            #mean_pixel_value = sum(sum(single_cell_gradient_array))/(len(single_cell_gradient_array[0])*len(single_cell_gradient_array))
            mean_pixel_value = numpy.mean(numpy.mean(single_cell_gradient_array))
            if mean_pixel_value > 25:
                old_cell_margin = cell_margin
                cell_margin = 0
                single_cell_input_array = input_array[start_on_row[cell_col] + cell_margin:start_on_row[cell_col]+cell_height-cell_margin,start_on_col+cell_margin:start_on_col+cell_width-cell_margin]
                #single_cell_im = Image.fromarray(single_cell_input_array)
                print(f"Identifying number on row {cell_row} and col {cell_col}")
                identify_number(single_cell_input_array, true_numbers)
                #single_cell_im.show()
                cell_margin = old_cell_margin

            #print(f"cell_row: {cell_row} cell_col: {cell_col} sum single_cell_gradient_array: {sum(sum(single_cell_gradient_array))/(len(single_cell_gradient_array[0])*len(single_cell_gradient_array))}")


            input_array[start_on_row[cell_col] + cell_margin:start_on_row[cell_col]+cell_height-cell_margin, start_on_col+cell_margin] = 255
            input_array[start_on_row[cell_col] + cell_margin:start_on_row[cell_col]+cell_height-cell_margin, start_on_col+cell_width-cell_margin] = 255
            input_array[start_on_row[cell_col] + cell_margin, start_on_col+cell_margin:start_on_col+cell_width-cell_margin] = 255
            input_array[start_on_row[cell_col]+cell_height-cell_margin, start_on_col+cell_margin:start_on_col+cell_width-cell_margin] = 255

            start_on_col += cell_width

            start_on_row[cell_col] += cell_height

        

    result_image = Image.fromarray(input_array)
    result_image.show()

def segment_cells2(normalized_array, threshold):
    
    width = len(normalized_array)
    cell_width = int(round(width/9))
    cell_margin = int(round(cell_width*0.12))

    #load list of thresholded images of the numbers to compare to:
    true_numbers = []
    for ii in range(1,10):
        path =  'numbers2/number_' + str(ii) + '.npy'
        with open(path,'rb') as f:
            true_numbers.append(numpy.load(f))

    true_gradient_numbers = []
    for ii in range(1,10):
        path =  'numbers2/gradient_' + str(ii) + '.npy'
        with open(path,'rb') as f:
            true_gradient_numbers.append(numpy.load(f))

        
    #number_im = Image.fromarray(true_gradient_numbers[3])
    #number_im.show()    


    for cell_row in range(0,9):
        for cell_col in range(0,9):
    
            current_cell = normalized_array[cell_row*cell_width+cell_margin:(cell_row+1)*cell_width-cell_margin, cell_col*cell_width+cell_margin:(cell_col+1)*cell_width-cell_margin]
            
            cell_mean = numpy.mean(numpy.mean(current_cell))
            
            current_cell = gaussian_blur(current_cell)
            current_cell_for_gradient = current_cell.copy()
            current_cell[current_cell>cell_mean*0.77] = 255
            current_cell[current_cell<=cell_mean*0.77]=0
            cell_sample_margin = int(round(cell_width*0.35))
            cell_sample = current_cell[cell_sample_margin:cell_width-cell_sample_margin, cell_sample_margin:cell_width-cell_sample_margin]
            #print(f"row: {cell_row} col: {cell_col} sample_mean: {numpy.mean(numpy.mean(cell_sample))}")
            if numpy.mean(numpy.mean(cell_sample))<250:
                start_row = 0
                current_row = current_cell[0,:]
                number_of_black_pixels = len(numpy.where(current_row<250)[0])
                while number_of_black_pixels < 2:
                    start_row += 1
                    current_row = current_cell[start_row, :]
                    number_of_black_pixels = len(numpy.where(current_row<250)[0])
                start_col = 0
                current_col = current_cell[:,0]
                number_of_black_pixels = len(numpy.where(current_col<250)[0])
                while number_of_black_pixels < 2:
                    start_col += 1
                    current_col = current_cell[:, start_col]
                    number_of_black_pixels = len(numpy.where(current_col<250)[0])
                #cut out for simple threshold
                normalized_cell = current_cell[start_row:start_row+24,start_col:start_col+20]

                #cut out for gradient threshold
                start_row = max(0, (start_row-2))
                start_col = max(0, (start_col-2))
                normalized_gradient_cell = current_cell_for_gradient[start_row:start_row+26, start_col:start_col+22]
                normalized_gradient_cell = sobel_convolution(normalized_gradient_cell)
                gradient_cell_im = Image.fromarray(normalized_gradient_cell)
                #gradient_cell_im.show()

                gradient_cell_mean = numpy.mean(numpy.mean(normalized_gradient_cell))
                normalized_gradient_cell[normalized_gradient_cell>gradient_cell_mean*0.85] = 255
                normalized_gradient_cell[normalized_gradient_cell<=gradient_cell_mean*0.85] = 0

                gradient_cell_im = Image.fromarray(normalized_gradient_cell)
                #gradient_cell_im.show()

                #The following code is only used when saving images of numbers for training etc. 
                save_number = False
                #save_number = True
                if save_number:
                    save_numpy_array(normalized_cell, f"numbers2/row_{cell_row}_col_{cell_col}.npy")
                    save_numpy_array(normalized_gradient_cell, f"numbers2/gradient_row_{cell_row}_col_{cell_col}.npy")
                    cell_im = Image.fromarray(normalized_cell)
                    gradient_cell_im = Image.fromarray(normalized_gradient_cell)
                    filename = f"numbers2/row_{cell_row}_col_{cell_col}.jpg"
                    gradient_filename = f"numbers2/gradient_row_{cell_row}_col_{cell_col}.jpg"
                    if cell_im.mode != 'RGB':
                        cell_im = cell_im.convert('RGB')
                    if gradient_cell_im.mode != 'RGB':
                        gradient_cell_im = gradient_cell_im.convert('RGB')

                    cell_im.save(filename)
                    gradient_cell_im.save(gradient_filename)
                    cell_im.show()
                    
                
                
                print(f"Trying to identify number in row: {cell_row} and col: {cell_col}")
                identify_number(normalized_cell, true_numbers)
                print(f"Trying to identify number via gradient in row: {cell_row} and col: {cell_col} ")
                identify_number(normalized_gradient_cell, true_gradient_numbers)
    

def identify_number(single_cell_input_array, true_numbers):
    #single_cell_input_array = [[0 if single_cell < 150 else 255 for single_cell in row] for row in single_cell_input_array]
    #single_cell_input_array = numpy.asarray(single_cell_input_array)
    
    height = len(single_cell_input_array)
    width = len(single_cell_input_array[0])

    sub_image_height = len(true_numbers[0])
    sub_image_width = len(true_numbers[0][0])

    if height != sub_image_height or width != sub_image_width:
        print(f"Error: image and sub_image dimensions don't match. Height: {height} sub_image_height: {sub_image_height} Width: {width} sub_image_width: {sub_image_width}")
    #start_row = int((height-sub_image_height)/2)
    #start_col = int((width-sub_image_width)/2)
    

    #print(f"height: {height} width: {width} sub_image_height: {sub_image_height} sub_image_width: {sub_image_width}")

    best_score = 100000000
    second_best_score = best_score
    guess = 0
    second_guess = 0

    #number_im = Image.fromarray(true_numbers[8])
    #number_im.show(title="Loaded sub_image")

    for number in range(len(true_numbers)):
        #number_im = Image.fromarray(true_numbers[number])
        #number_im.show(title="Loaded sub_image")
        
        difference_matrix = abs(single_cell_input_array-true_numbers[number])
        difference = sum(sum(difference_matrix))

        #print(f"true number: {number+1} difference: {difference}")

        if difference < best_score:
            second_best_score = best_score
            second_guess = guess
            best_score = difference
            guess = number+1
            #print(f"New guess: {guess} difference: {difference} and prev best_score: {second_best_score}")
        elif difference < second_best_score:
            #print(f"New second_guess without first guess: {number+1} difference: {difference} and prev second_best_score: {second_best_score}")

            second_best_score = difference
            second_guess = number+1
        
    
    
    print(f"Guess: {guess} and second guess: {second_guess}")


    #result_im = Image.fromarray(sub_image)
    #result_im.show(title="Loaded sub_image")


    #single_cell_im = Image.fromarray(single_cell_input_array)
    #single_cell_im.show()
       
