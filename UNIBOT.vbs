' Avvia UniBot senza mostrare la finestra del terminale.
Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")
scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
shell.Run """" & scriptDir & "\start_chatbot.bat""", 0, False
