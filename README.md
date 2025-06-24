# TC7Agentes

Trabajo Corto 7 de Agentes

# Video demostración

El enlace al video es el siguiente:
[Enlace de video](https://youtu.be/TtJsEoA0Vro)

https://youtu.be/TtJsEoA0Vro

# Enlace al repositorio del proyecto

[Enlace al repositorio](https://github.com/Dandiego235/TC7Agentes)

https://github.com/Dandiego235/TC7Agentes

# Guía de usuario

## Variables de entorno

Es necesario especificar las variables de entorno en un archivo `.env`. Estas son:

```
OPENAI_API_KEY=
TAVILY_API_KEY=
USER_AGENT = "LangchainAgent/1.0"
```

## Chat Web

Para correr el webchat, es necesario instalar los requerimientos en `requirements.txt`. Luego, se debe correr el archivo `webChat.py`. Esto abre la página en el `localhost:5000`. Ya con esto, se puede hablar con el agente educativo TECGPT. Estas utilizan el HTML de `templates/chat.html` y los estilos de `static.css`.

## agent.py

Este archivo es el que contiene la configuración de los agentes. Está la configuración del supervisor, el agente de matemáticas, el de ciencia, historia y de gramática. Las imágenes en el fólder de images son los diagramas de los agentes.

## rag.py

Este archivo contiene tiene las funcionalidades para implementar el Retrieval Augmented Generation (RAG). Procesa PDF's, lee páginas web y contiene el agente RAG.
