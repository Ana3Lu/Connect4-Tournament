# Connect4-Tournament

Repositorio del proyecto final del curso Fundamentos de IA, cuyo objetivo es que cada integrante diseñe e implemente un agente capaz de jugar Connect-4 de forma autónoma, aplicando técnicas de inteligencia artificial vistas durante el curso.

## Estructura del repositorio

La carpeta `connect4` contiene el motor del juego, las reglas, los tipos de datos y las utilidades compartidas por todos los agentes. Por su parte, la carpeta `groups` aloja las políticas de cada participante, organizadas en subcarpetas individuales. Los resultados de las partidas se guardan automáticamente en formato JSON dentro de la carpeta `versus`, mientras que `main.py` y `tournament.py` corresponden al punto de entrada y a la lógica del torneo respectivamente.

## Cómo ejecutar el torneo

Se requiere Python 3.10 o superior. Una vez clonado el repositorio, basta con ejecutar el archivo principal desde la raíz del proyecto:

    python main.py

Los resultados de cada partida quedarán guardados automáticamente en la carpeta `versus`.

## Dependencias

El proyecto utiliza principalmente numpy para el manejo del tablero. A medida que cada agente se desarrolle, es posible que se incorporen librerías adicionales para entrenamiento, visualización de resultados y análisis de métricas, como matplotlib o seaborn. Cada estudiante especificará en su propio readme las dependencias particulares de su implementación.

## Contribución por estudiante

Cada integrante trabaja en su propia rama y mantiene su agente dentro de una subcarpeta personal en `groups`. Al finalizar el desarrollo, la política final se integra a la rama principal para ser evaluada en el torneo grupal.
