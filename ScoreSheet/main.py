from preprocessing import preprocess_image
from move_extraction import extract_move_boxes
from postprocessing import error_correct, make_pgn, make_lichess_url
from ocr import OCRModel

def main():

    # Preprocess the image
    image_path = 'example2.png'
    processed_image = preprocess_image(image_path)

    # Extract move boxes from the preprocessed images (as ROIs)
    move_boxes = extract_move_boxes(processed_image)

    # Initialize single-character OCR model
    ocr_model = OCRModel('Best_points.h5')

    # Perform OCR on each move box; 'perform_ocr' returns top N predictions in a tuple of candidates
    move_candidates = [ocr_model.perform_ocr(box, n=10) for box in move_boxes]

    move_candidates = [candidate for candidate in move_candidates if candidate != tuple()]

    print(len(move_candidates))
    # Error correct based on chess rules
    moves = error_correct(move_candidates)

    print(moves)

    pgn = make_pgn(moves)

    print(pgn)

    url = make_lichess_url(moves)

    print(url)


    

if __name__ == "__main__":
    main()