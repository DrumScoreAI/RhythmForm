# RhythmForm Usage Examples

## **Important: Ethical Use and Licensing**

RhythmForm is a tool to digitize drum scores from PDF files. It is designed for use with sheet music that you have created yourself or for which you have obtained the legal rights to process.

**It is your sole responsibility to ensure you are not infringing on any copyrights.** The developers of RhythmForm do not condone the use of this software to process copyrighted materials without the proper license or permission from the copyright holder.

---

## Primary Goal: PDF to MusicXML Conversion

The main purpose of RhythmForm is to take a PDF file of a drum score and convert it into a standard, interactive MusicXML file.

### The `convert` Command

The primary command for this process is `convert`.

**Syntax:**
```bash
rhythmform convert <path/to/input.pdf> --output <path/to/output.xml>
```

-   `<path/to/input.pdf>`: The PDF file of the drum score you want to process.
-   `--output <path/to/output.xml>`: The name and location for the generated MusicXML file.

## Example Workflow

Let's say you have a legally-owned PDF of a drum chart named `funky-groove.pdf` in your `Documents` folder.

### 1. Run the Conversion
To convert it, you would run the following command:
```bash
rhythmform convert "~/Documents/funky-groove.pdf" --output "funky-groove.xml"
```

### 2. What Happens Behind the Scenes
1.  **Image Processing**: RhythmForm loads the PDF and converts each page into a high-resolution image.
2.  **AI Recognition**: The trained RhythmForm transformer model analyzes each page image, translating the visual notation into a symbolic music text (SMT) representation.
3.  **Score Construction**: The SMT data is then used to programmatically build a `music21` score object, creating all the notes, rests, chords, and measures.
4.  **File Generation**: The final score object is exported as a standard MusicXML file.

### 3. Understanding the Output
A successful conversion will look like this in your terminal:
```
📄 Loading PDF: ~/Documents/funky-groove.pdf
   - Found 2 pages.
🤖 Processing page 1/2 with RhythmForm model...
🤖 Processing page 2/2 with RhythmForm model...
🎼 Assembling MusicXML structure...
✅ Success! MusicXML file saved to: funky-groove.xml
```

### 4. Using the Output File
You can now open `funky-groove.xml` in any compatible music notation software, such as:
-   MuseScore Studio
-   GarageBand
-   Logic Pro X
-   Sibelius
-   Finale

In this software, you can play back the score, edit notes, change the tempo, and use it as an interactive practice tool.

## Troubleshooting

### Error: "Model checkpoint not found"
This means the trained model file (`.pth`) is not available. You may need to train the model first or download a pre-trained checkpoint. See the main `README.md` for details on training.

### Low-Quality Output
The accuracy of the conversion depends heavily on the quality of the source PDF. For best results, use high-resolution, cleanly scanned, or digitally-generated PDFs. Handwritten or blurry scores may produce poor results.