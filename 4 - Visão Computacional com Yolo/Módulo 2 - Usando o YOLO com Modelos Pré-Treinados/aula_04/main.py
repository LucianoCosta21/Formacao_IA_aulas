from ultralytics import YOLO
import cv2
import pandas as pd

model = YOLO('yolo11n.pt')

"""
## filtrar por classes e ajuste de nivel de confiança
results = model.predict(source="imagem.jpg", conf=0.7, classes=[2])
results[0].show()
"""

class_colors = {
    0: (0, 0, 255),   # vermelho para pessoa
    2: (0, 255, 0)    # verde para carro
}

default_color = (255,0,0)

image_path = "imagem.jpg"
frame = cv2.imread(image_path)
results = model(frame)[0]

for box in results.boxes:

    class_id = int(box.cls[0])
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    confidence = float(box.conf[0])

    color = class_colors.get(class_id, default_color)

    label = model.names[class_id] 
    
    text = f"{label} {confidence:.2f}"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

cv2.imshow("Cores por Classe", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Para salvar com um nome personalizado
results[0].save(filename='detec_imagem.jpg')

## importar resultado para cvs
data = results[0].boxes.data.cpu().numpy()
df = pd.DataFrame(data, columns=['x1', 'y1', 'x2', 'y2', 'conf', 'class'])
df.to_csv('detec_resultados.csv', index=False)