# Estimació de l'ocupació de les platges

J. F. Sánchez García, P. Bibiloni, and A. Oliver Tomàs, “Automatic Crowd Counting on the Beaches of the Balearic Islands Using Convolutional Neural Networks,” Frontiers in Artificial Intelligence and Applications. IOS Press, Sept. 22, 2025. doi: [10.3233/faia250604](http://dx.doi.org/10.3233/FAIA250604).

Based on the Final Degree Project of Juan Francisco Sánchez García “Monitoreo automático de la ocupación de las playas de las Islas Baleares”, defended at Universitat de les Illes Balears in the 2023-24 academic year [[GitHub repo](https://github.com/PBibiloni/beachcamweb)].

This work is supported by the project PID2020-113870GB-I00 – “Desarrollo de herramientas de Soft Computing para la Ayuda al Diagnóstico Clínico y a la Gestión deEmergencias (HESOCODICE)” funded by MICIU/AEI/10.13039/501100011033/.

## Installation

### Virtual environment

Virtual environment `.venv` encouraged.

### Dependencies

```
pip install -r requirements.txt
```

### Configuration

Copy `config/local_settings.example.py` into `config/local_settings.py` and edit its fields.

## Development

### Translation

#### Per a detectar noves strings a traduir:

```
python manage.py makemessages -all
```

#### Per a compilar els .po a .mo

```
python.exe manage.py compilemessages --ignore .venv
```