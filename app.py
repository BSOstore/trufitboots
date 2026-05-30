from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
from vision import process_hoof_image
import sqlite3

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
PROCESSED_FOLDER = "static/processed"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)


def get_db_connection():
    conn = sqlite3.connect("trufitboots.db")
    conn.row_factory = sqlite3.Row
    return conn


def get_best_match(length, width):

    target_length = length + 7
    target_width = width + 7

    conn = get_db_connection()

    result = conn.execute(
        """
        SELECT *,
        ABS(length - ?) + ABS(width - ?) AS score
        FROM "TruFit Products"
        ORDER BY score ASC
        LIMIT 1
    """,
        (target_length, target_width),
    ).fetchone()

    conn.close()

    return result


@app.route("/debug-columns")
def debug_columns():

    conn = get_db_connection()

    result = conn.execute("""
        PRAGMA table_info("TruFit Products")
    """).fetchall()

    conn.close()

    output = ""

    for row in result:
        output += f"{row[1]}<br>"

    return output


@app.route("/test-db")
def test_db():
    conn = get_db_connection()

    products = conn.execute('SELECT * FROM "TruFit Products"').fetchall()

    conn.close()

    output = ""

    for product in products:
        output += f"{product['length']} x {product['width']}<br>"

    return output


@app.route("/products")
def products():

    conn = get_db_connection()

    products = conn.execute('SELECT * FROM "TruFit Products"').fetchall()

    conn.close()

    return render_template("products.html", products=products)


@app.route("/match/<int:length>/<int:width>")
def match(length, width):

    conn = get_db_connection()

    result = conn.execute(
        """
        SELECT *,
        ABS(length - ?) + ABS(width - ?) AS score
        FROM "TruFit Products"
        ORDER BY score ASC
        LIMIT 1
    """,
        (length, width),
    ).fetchone()

    conn.close()

    if result:
        return f"""
        Best Match:<br>
        Length: {result['length']}<br>
        Width: {result['width']}<br>
        Score: {result['score']}
        """

    return "No match found."


def get_insert_library():
    image_folder = os.path.join("static", "images")
    inserts = []

    if os.path.exists(image_folder):
        for file in os.listdir(image_folder):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                name = os.path.splitext(file)[0]

                try:
                    size_part = name[:-1]
                    direction = name[-1].upper()

                    length, width = size_part.split("x")

                    inserts.append(
                        {
                            "filename": file,
                            "length": int(length),
                            "width": int(width),
                            "direction": direction,
                        }
                    )

                except:
                    pass

    return inserts


@app.route("/", methods=["GET", "POST"])
def index():
    original_image = None
    processed_image = None
    message = None
    measured_length = None
    measured_width = None
    best_match = None
    length_insert = None
    width_insert = None

    if request.method == "POST":
        file = request.files.get("hoof_image")

        if not file or file.filename == "":
            message = "No image selected."
        else:
            filename = secure_filename(file.filename)

            upload_path = os.path.join(UPLOAD_FOLDER, filename)
            processed_filename = f"processed_{filename}"
            processed_path = os.path.join(PROCESSED_FOLDER, processed_filename)

            file.save(upload_path)

            success, message, measured_length, measured_width = process_hoof_image(
                upload_path, processed_path
            )

            if success:
                original_image = upload_path
                processed_image = processed_path

                best_match = get_best_match(measured_length, measured_width)

                recommended_length = best_match["length"]
                recommended_width = best_match["width"]

                length_insert = f"{recommended_length}x{recommended_width}L.jpg"
                width_insert = f"{recommended_length}x{recommended_width}W.jpg"

    return render_template(
        "index.html",
        original_image=original_image,
        processed_image=processed_image,
        message=message,
        measured_length=measured_length,
        measured_width=measured_width,
        best_match=best_match,
        length_insert=length_insert,
        width_insert=width_insert,
        images=get_insert_library(),
    )


if __name__ == "__main__":
    app.run(debug=True)
