import os
import subprocess
from flask import Flask, render_template, request, redirect, url_for, flash, send_file, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.pdfgen import canvas
import tempfile
import shutil
import cv2
import numpy as np

# Sprawdź i utwórz katalog instance jeśli nie istnieje
instance_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'instance')
if not os.path.exists(instance_path):
    os.makedirs(instance_path)

app = Flask(__name__, instance_path=instance_path)
app.secret_key = 'supersecretkey'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['PERMANENT_SESSION_LIFETIME'] = 1800  # 30 minut
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SECURE'] = False  # Zmienić na True w środowisku produkcyjnym

# Inicjalizacja sesji Flask
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = os.path.join(instance_path, 'flask_session')

# Sprawdź czy katalog sesji istnieje i jest pusty
if os.path.exists(app.config['SESSION_FILE_DIR']):
    # Usuń wszystkie pliki sesji
    for file in os.listdir(app.config['SESSION_FILE_DIR']):
        file_path = os.path.join(app.config['SESSION_FILE_DIR'], file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Błąd podczas czyszczenia sesji: {e}")
else:
    # Jeśli katalog nie istnieje, utwórz go
    os.makedirs(app.config['SESSION_FILE_DIR'])

# Sprawdź i utwórz katalog uploaded_files jeśli nie istnieje
UPLOAD_FOLDER = 'uploaded_files'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

ALLOWED_EXTENSIONS = {'jpg', 'png', 'pdf', 'docx', 'odt', 'tiff', 'bmp', 'gif', 'pdf', 'epub', 'xlsx', 'pptx', 'mp3', 'wav', 'flac', 'mp4', 'avi', 'mov', 'mkv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
EXIFTOOL_PATH = r"D:\LOAD\LOAD_APP_v2.1\uploaded_files\exiftool.exe"

# Sprawdzenie czy plik ExifTool istnieje
if not os.path.exists(EXIFTOOL_PATH):
    print(f"Błąd: Plik ExifTool nie istnieje w ścieżce: {EXIFTOOL_PATH}")
    exit(1)

# Sprawdzenie czy plik jest plikiem wykonywalnym
if not os.access(EXIFTOOL_PATH, os.X_OK):
    print(f"Błąd: Plik ExifTool nie ma uprawnień do wykonywania: {EXIFTOOL_PATH}")
    exit(1)

db = SQLAlchemy(app)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash("Nie wybrano pliku.", "danger")
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash("Nie wybrano pliku.", "danger")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            if not os.path.exists(filepath):
                flash(f"Plik {filepath} nie istnieje lub jest niedostępny.", "danger")
                return redirect(url_for('analyze'))
            try:
                # Najpierw sprawdź metadane
                process = subprocess.run(
                    [EXIFTOOL_PATH, filepath],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding='utf-8'
                )
                if process.returncode != 0:
                    flash(f"Błąd odczytu metadanych: {process.stderr}", "danger")
                    return redirect(url_for('analyze'))
                
                metadata = process.stdout.splitlines()
                origin_analysis = analyze_image_origin(filepath)
                
                # Zapisz metadane do sesji
                session['metadata'] = metadata
                session['filename'] = filename
                session['filepath'] = filepath
                
                return render_template('report.html', filename=filename, metadata=metadata, origin_analysis=origin_analysis)
            except Exception as e:
                flash(f"Błąd podczas analizy pliku: {e}", "danger")
                return redirect(url_for('analyze'))
        else:
            flash("Niedozwolony typ pliku.", "danger")
            return redirect(request.url)
    return render_template('analyze.html')

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import cv2
import numpy as np

def analyze_image_origin(filepath):
    try:
        image = cv2.imread(filepath)
        if image is None:
            return "Nie udało się wczytać obrazu."
        blur = cv2.Laplacian(image, cv2.CV_64F).var()
        edges = cv2.Canny(image, 100, 200)
        edge_density = np.sum(edges) / (image.shape[0] * image.shape[1])
        if blur < 100 or edge_density > 0.3:
            return "Obraz prawdopodobnie wygenerowany przez AI."
        else:
            return "Obraz wygląda na wykonany przez człowieka."
    except Exception as e:
        return f"Błąd analizy obrazu: {e}"



@app.route('/remove-metadata/', methods=['GET'])
def remove_metadata():
    filename = request.args.get('filename')
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        flash(f"Plik {filepath} nie istnieje.", "danger")
        return redirect(url_for('analyze'))
    try:
        clean_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"clean_{filename}")
        process = subprocess.run(
            [EXIFTOOL_PATH, "-all=", "-overwrite_original", filepath],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8'
        )
        if process.returncode == 0:
            flash("Metadane zostały pomyślnie usunięte.", "success")
        else:
            flash(f"Błąd podczas usuwania metadanych: {process.stderr}", "danger")
        return redirect(url_for('report', filename=filename))
    except Exception as e:
        flash(f"Błąd podczas usuwania metadanych: {e}", "danger")
        return redirect(url_for('analyze'))

@app.route('/download-txt/', methods=['GET'])
def download_txt():
    filename = request.args.get('filename')
    if not filename:
        flash("Nie podano nazwy pliku", "danger")
        return redirect(url_for('report', filename=filename))
    
    # Sprawdź czy metadane są w sesji
    if 'metadata' not in session:
        flash("Brak metadanych w sesji", "danger")
        return redirect(url_for('report', filename=filename))
    
    # Użyj metadanych z sesji
    metadata = session['metadata']
    filepath = session.get('filepath')
    
    try:
        # Tworzenie pliku TXT
        txt_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{filename}_report.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("# Raport Metadanych\n\n")
            f.write(f"Plik: {filename}\n\n")
            f.write("=== Metadane ===\n\n")
            for line in metadata:
                f.write(f"- {line}\n")
        
        if not os.path.exists(txt_path):
            flash("Błąd: TXT nie został utworzony", "danger")
            return redirect(url_for('report', filename=filename))
            
        return send_file(txt_path, as_attachment=True, download_name=f"{filename}_report.txt")
    except Exception as e:
        print(f"Błąd podczas generowania TXT: {str(e)}")
        flash(f"Błąd podczas generowania TXT: {str(e)}", "danger")
        return redirect(url_for('report', filename=filename))

@app.route('/metadata-tools/', methods=['GET', 'POST'])
def metadata_tools():
    filename = request.args.get('filename')
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    clean_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"clean_{filename}")
    
    if not os.path.exists(filepath):
        flash(f"Plik {filepath} nie istnieje.", "danger")
        return redirect(url_for('analyze'))
    
    if request.method == 'POST':
        action = request.form.get('action')
        category = request.form.get('category')
        try:
            if action == "remove_all":
                process = subprocess.run(
                    [EXIFTOOL_PATH, "-all=", "-overwrite_original", filepath],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding='utf-8'
                )
                if process.returncode != 0:
                    flash(f"Błąd podczas usuwania metadanych: {process.stderr}", "danger")
                    return redirect(url_for('metadata_tools', filename=filename))
                flash("Wszystkie metadane zostały usunięte.", "success")
                return redirect(url_for('metadata_tools', filename=filename))
            elif action == "remove_category":
                if category:
                    process = subprocess.run(
                        [EXIFTOOL_PATH, f"-{category}=" , "-overwrite_original", filepath],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        encoding='utf-8'
                    )
                    if process.returncode != 0:
                        flash(f"Błąd podczas usuwania kategorii: {process.stderr}", "danger")
                        return redirect(url_for('metadata_tools', filename=filename))
                    flash(f"Kategoria {category} została usunięta.", "success")
                    return redirect(url_for('metadata_tools', filename=filename))
                else:
                    flash("Nie wybrano kategorii do usunięcia.", "warning")
                    return redirect(url_for('metadata_tools', filename=filename))
        except Exception as e:
            flash(f"Błąd podczas przetwarzania metadanych: {str(e)}", "danger")
            return redirect(url_for('metadata_tools', filename=filename))
    else:
        # Pobierz metadane
        try:
            process = subprocess.run(
                [EXIFTOOL_PATH, filepath],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8'
            )
            if process.returncode != 0:
                flash(f"Błąd podczas odczytu metadanych: {process.stderr}", "danger")
                metadata = []
            else:
                metadata = process.stdout.split('\n')
                metadata = [line.strip() for line in metadata if line.strip()]
        except Exception as e:
            flash(f"Błąd podczas odczytu metadanych: {str(e)}", "danger")
            metadata = []
        
        # Sprawdź czy plik czysty istnieje
        if os.path.exists(clean_filepath):
            clean_process = subprocess.run(
                [EXIFTOOL_PATH, clean_filepath],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8'
            )
            clean_metadata = clean_process.stdout.split('\n')
            clean_metadata = [line.strip() for line in clean_metadata if line.strip()]
        else:
            clean_metadata = []
        
        return render_template(
            'metadata_tools.html',
            filename=filename,
            metadata=metadata,
            clean_metadata=clean_metadata
        )
    
    if request.method == 'POST':
        action = request.form.get('action')
        category = request.form.get('category')
        try:
            if action == "remove_all":
                process = subprocess.run(
                    [EXIFTOOL_PATH, "-all=", "-overwrite_original", filepath],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding='utf-8'
                )
                if process.returncode != 0:
                    flash(f"Błąd podczas usuwania metadanych: {process.stderr}", "danger")
                    return redirect(url_for('metadata_tools', filename=filename))
                flash("Wszystkie metadane zostały usunięte.", "success")
                return redirect(url_for('metadata_tools', filename=filename))
            elif action == "remove_category" and category:
                process = subprocess.run(
                    [EXIFTOOL_PATH, f"-{category}=" , "-overwrite_original", filepath],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding='utf-8'
                )
                if process.returncode != 0:
                    flash(f"Błąd podczas usuwania kategorii: {process.stderr}", "danger")
                    return redirect(url_for('metadata_tools', filename=filename))
                flash(f"Kategoria {category} została usunięta.", "success")
                return redirect(url_for('metadata_tools', filename=filename))
        except Exception as e:
            flash(f"Błąd podczas przetwarzania metadanych: {str(e)}", "danger")
            return redirect(url_for('metadata_tools', filename=filename))
    else:
        try:
            process = subprocess.run(
                [EXIFTOOL_PATH, filepath],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8'
            )
            if process.returncode != 0:
                flash(f"Błąd podczas odczytu metadanych: {process.stderr}", "danger")
                metadata = []
            else:
                metadata = process.stdout.split('\n')
                metadata = [line.strip() for line in metadata if line.strip()]
        except Exception as e:
            flash(f"Błąd podczas odczytu metadanych: {str(e)}", "danger")
            metadata = []
        
        # Sprawdź czy plik czysty istnieje
        if os.path.exists(clean_filepath):
            clean_process = subprocess.run(
                [EXIFTOOL_PATH, clean_filepath],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8'
            )
            clean_metadata = clean_process.stdout.split('\n')
            clean_metadata = [line.strip() for line in clean_metadata if line.strip()]
        else:
            clean_metadata = []
        
        return render_template(
            'metadata_tools.html',
            filename=filename,
            metadata=metadata,
            clean_metadata=clean_metadata
        )

@app.route('/download-pdf/', methods=['GET'])
def download_pdf():
    filename = request.args.get('filename')
    if not filename:
        flash("Nie podano nazwy pliku", "danger")
        return redirect(url_for('report', filename=filename))
    
    # Sprawdź czy metadane są w sesji
    if 'metadata' not in session:
        flash("Brak metadanych w sesji", "danger")
        return redirect(url_for('report', filename=filename))
    
    # Użyj metadanych z sesji
    metadata = session['metadata']
    
    try:
        # Tworzenie tymczasowego pliku PDF
        temp_path = tempfile.mkdtemp()
        pdf_path = os.path.join(temp_path, f"{filename}_report.pdf")
        
        # Ustawienia dokumentu
        doc = SimpleDocTemplate(pdf_path, pagesize=A4,
                               rightMargin=72, leftMargin=72,
                               topMargin=72, bottomMargin=72)
        
        # Stylizacja
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name='Metadata', fontSize=10, leading=14))
        styles.add(ParagraphStyle(name='Title', fontSize=24, leading=24, alignment=TA_CENTER))
        styles.add(ParagraphStyle(name='Subtitle', fontSize=16, leading=16))
        
        # Treść dokumentu
        elements = []
        
        # Tytuł
        elements.append(Paragraph("Raport Metadanych", styles['Title']))
        elements.append(Spacer(1, 20))
        
        # Nazwa pliku
        elements.append(Paragraph(f"Plik: {filename}", styles['Subtitle']))
        elements.append(Spacer(1, 12))
        
        # Linia podziału
        elements.append(Spacer(1, 12))
        elements.append(Paragraph("-" * 60, styles['Normal']))
        elements.append(Spacer(1, 20))
        
        # Sekcja metadanych
        elements.append(Paragraph("Metadane:", styles['Normal']))
        elements.append(Spacer(1, 12))
        
        # Tabela metadanych
        data = []
        for line in metadata:
            data.append([Paragraph(line, styles['Metadata'])])
        
        table = Table(data)
        table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12)
        ]))
        
        elements.append(table)
        
        # Generowanie PDF
        doc.build(elements)
        
        # Sprawdzenie czy plik został utworzony
        if not os.path.exists(pdf_path):
            flash("Błąd: PDF nie został utworzony", "danger")
            return redirect(url_for('report', filename=filename))
            
        # Wysłanie pliku
        return send_file(pdf_path, as_attachment=True, download_name=f"{filename}_report.pdf")
        
    except Exception as e:
        print(f"Błąd podczas generowania PDF: {str(e)}")
        flash(f"Błąd podczas generowania PDF: {str(e)}", "danger")
        return redirect(url_for('report', filename=filename))
    finally:
        # Usunięcie tymczasowego katalogu
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path)
            print(f"Tymczasowy katalog {temp_path} został usunięty")
        flash("Plik nie istnieje.", "danger")
        return redirect(url_for('analyze'))
    try:
        process = subprocess.run(
            [EXIFTOOL_PATH, "-all=", f"-comment=LENS OSINT ANALYZER DATA", "-overwrite_original", original_filepath],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8'
        )
        if process.returncode != 0:
            flash(f"Błąd podczas przetwarzania pliku: {process.stderr}", "danger")
            return redirect(url_for('metadata_tools', filename=filename))
        return send_file(original_filepath, as_attachment=True)
    except Exception as e:
        flash(f"Błąd podczas pobierania pliku: {e}", "danger")
        return redirect(url_for('metadata_tools', filename=filename))

@app.route('/report')
def report():
    reports = []
    return render_template('report.html', reports=reports)

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        try:
            os.makedirs(UPLOAD_FOLDER)
            print(f"Utworzono folder {UPLOAD_FOLDER}")
        except Exception as e:
            print(f"Błąd podczas tworzenia folderu {UPLOAD_FOLDER}: {e}")
            exit(1)
    
    # Sprawdzenie uprawnień do folderu
    try:
        test_file = os.path.join(UPLOAD_FOLDER, 'test.txt')
        with open(test_file, 'w') as f:
            f.write('Test')
        os.remove(test_file)
        print(f"Folder {UPLOAD_FOLDER} ma poprawne uprawnienia")
    except Exception as e:
        print(f"Błąd uprawnień do folderu {UPLOAD_FOLDER}: {e}")
        exit(1)
    
    app.run(debug=True)
