def get_perimeter_pixels(image):
    perimeter_pixels = []
    x, y = image.shape
    for (count, i) in enumerate(image):
        for j, j_val in enumerate(i):
            if(count == 0) or (count == x-1) or (j == 0) or (j == y-1):
                perimeter_pixels.append(j_val)
    return(perimeter_pixels)