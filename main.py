from fusion_module.stress_fusion import stress_level

face_emotion = "angry"   # from face_realtime
voice_emotion = "fear"   # from voice_realtime

final_stress = stress_level(face_emotion, voice_emotion)

print("Face Emotion:", face_emotion)
print("Voice Emotion:", voice_emotion)
print("Worker Stress Level:", final_stress)
