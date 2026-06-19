import asyncio
import os
import threading
import time

import pygame
import edge_tts

VOICE = "es-ES-ElviraNeural"   # Microsoft Neural TTS — español, voz femenina natural
CACHE = ".voice_cache"

PHRASES = {
    "inicio":        "Comencemos. Posición inicial.",
    "buen_rep":      "¡Muy bien!",
    "baja_mas":      "Baja un poco más.",
    "espalda_recta": "Mantén la espalda más recta.",
    **{f"rep_{i}": str(i) for i in range(1, 21)},
}


class VoiceCoach:
    def __init__(self):
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        self.playing = False

        os.makedirs(CACHE, exist_ok=True)
        self._pregenerate()

    def _pregenerate(self):
        """Genera los MP3 la primera vez (necesita internet una sola vez)."""
        pending = [k for k in PHRASES if not os.path.exists(self._path(k))]
        if not pending:
            return

        print(f"Generando {len(pending)} archivos de voz (solo la primera vez)...")
        asyncio.run(self._generate_all(pending))
        print("Voz lista.")

    async def _generate_all(self, keys):
        for key in keys:
            print(f"  {key}...")
            await edge_tts.Communicate(PHRASES[key], VOICE).save(self._path(key))

    def _path(self, key):
        return os.path.join(CACHE, f"{key}.mp3")

    def say(self, key):
        if self.playing:
            return
        path = self._path(key)
        if os.path.exists(path):
            threading.Thread(target=self._play, args=(path,), daemon=True).start()

    def _play(self, path):
        self.playing = True
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.05)
        self.playing = False
