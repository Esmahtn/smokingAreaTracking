import os
import db_manager

if __name__ == '__main__':
    print('Initializing DB...')
    db_manager.init_db()

    # Ensure static/violations exists and create a dummy image
    os.makedirs('static/violations', exist_ok=True)
    img_path = os.path.join('static', 'violations', 'test_violation.jpg')
    try:
        with open(img_path, 'wb') as f:
            f.write(b'\xff\xd8\xff\xd9')  # minimal JPEG markers
    except Exception as e:
        print('Could not write dummy image:', e)

    print('Adding violation...')
    rec = db_manager.add_violation(person_id=9999, duration=42, image_path=img_path)
    print('Added:', rec)

    print('\nLast violations:')
    rows = db_manager.get_violations(5)
    for r in rows:
        print(r)

    # Cleanup: remove the dummy image file and delete inserted record
    try:
        os.remove(img_path)
    except Exception:
        pass

    print('\nTest complete')
