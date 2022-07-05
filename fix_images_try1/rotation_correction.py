
import piexif


def correct_rotation(img):
    """ correct image rotation based on exif data, such as in pics by mobile cameras """
    
    exif_bytes = None

    if "exif" in img.info:
    
        try:
            exif_dict = piexif.load(img.info['exif'])
            
            if piexif.ImageIFD.Orientation in exif_dict['0th']:

                if exif_dict['0th'][piexif.ImageIFD.Orientation] == 3: #upside down
                    img = img.rotate(180, expand=True)
                    exif_dict['0th'][piexif.ImageIFD.Orientation] = 1 # upright 
                    exif_dict['Exif'][41729] = b'1'
                    exif_bytes = piexif.dump(exif_dict)

                elif exif_dict['0th'][piexif.ImageIFD.Orientation] == 8: # img is turned cw
                    img = img.rotate(90, expand=True)
                    exif_dict['0th'][piexif.ImageIFD.Orientation] = 1
                    exif_dict['Exif'][41729] = b'1'
                    exif_bytes = piexif.dump(exif_dict)  

                elif exif_dict['0th'][piexif.ImageIFD.Orientation] == 6: # img is turned ccw
                    img = img.rotate(-90, expand=True)
                    exif_dict['0th'][piexif.ImageIFD.Orientation] = 1
                    exif_dict['Exif'][41729] = b'1'
                    exif_bytes = piexif.dump(exif_dict) 

                else:
                    exif_bytes = img.info['exif']
            
        except:
            print('broken exif')

        
    return img, exif_bytes
