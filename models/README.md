# Modèles

Les poids des modèles (`*.pt`) ne sont pas versionnés dans le dépôt. Il en faut trois :

| Fichier | Tâche | Classes |
|---|---|---|
| `detect_bubbles.pt` | Détection YOLOv8 | 0 = bulle avec queue, 1 = bulle sans queue, 2 = narration |
| `segment_bubbles.pt` | Segmentation YOLOv8 | 0 = bulle (contour complet, queue comprise) |
| `detect_characters.pt` | Détection YOLOv8 | une classe par personnage (propre à chaque série) |

## Entraîner ses propres modèles

1. Annoter des pages (par ex. avec Roboflow) en respectant les classes ci-dessus.
2. Exporter au format YOLOv8.
3. Entraîner :
   ```bash
   yolo detect train data=data.yaml model=yolov8n.pt imgsz=640 epochs=100
   yolo segment train data=data.yaml model=yolov8n-seg.pt imgsz=320 epochs=100
   ```
4. Copier les `best.pt` obtenus ici sous les noms ci-dessus.

Le modèle de personnages dépend de la série : penser à renseigner `yolo_class`
pour chaque personnage dans `config/characters.json`.
