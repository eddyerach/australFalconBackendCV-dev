# main.py
import sys
import os

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from src.vision import BunchDetector, GrapesDetector                          #Modulo vision (detectron)
from src.prediccion import Predictor                                          #Modulo prediccion (pytorch)


def main():
    #Cargar imagen
    model_bunches = 'models/model_final.pth'
    model_grapes  = 'models/model_v3.pth'
    input_path = '12C.JPG'

    #Instanciar modulos vision: BunchDetector y GrapesDetector
    bunch_detector = BunchDetector(model_bunches, input_path)
    bunch_detector.load_model()
    bunch_masks = bunch_detector.detect_bunches()
    grapes_detector = GrapesDetector(model_grapes, bunch_masks, input_path)
    grapes_detector.load_model()
    det, idx_comp_sol, idx_comp_no_sol = grapes_detector.detect_grapes()
    #print(f'In main: det: {det}, idx_comp_sol: {idx_comp_sol}, idx_comp_no_sol: {idx_comp_no_sol}')
    
    #Codigo predictor
    scaler = 'src/scaler.pkl'
    model_pred_path = 'src/aug4estado1234_man2_oct4.pth'
    predictor = Predictor(scaler, model_pred_path)
    pred = predictor.predict([idx_comp_sol, idx_comp_no_sol, det,
                  idx_comp_sol, idx_comp_no_sol, det,
                  idx_comp_sol, idx_comp_no_sol, det])
    print(f'pred: {pred}')

  
if __name__ == "__main__":
    main()
