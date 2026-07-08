# Zusammenarbeit am IEEE-LaTeX-Paper

Diese Anleitung beschreibt zwei praktikable Wege, um gemeinsam am Paper in `Docs/paper` zu arbeiten:

1. Overleaf fuer gemeinsame Online-Bearbeitung.
2. Lokales Arbeiten in VS Code mit PDF-Vorschau.

## Kurzfazit

Ja, das Paper kann mit wenig Aufwand in Overleaf genutzt werden. Die aktuelle Struktur ist einfach:

- `main.tex` ist die Hauptdatei.
- `references.bib` enthaelt die Literaturquellen.
- `figures/` enthaelt alle Abbildungen.

Nicht wichtig fuer Overleaf oder lokale Mitarbeit sind die automatisch erzeugten Build-Dateien wie `main.aux`, `main.bbl`, `main.blg`, `main.fls`, `main.fdb_latexmk`, `main.log`, `main.pdf` und der Ordner `preview/`. Diese Dateien koennen lokal neu erzeugt werden und muessen normalerweise nicht aktiv bearbeitet werden.

## Variante A: Overleaf verwenden

Diese Variante ist am einfachsten, wenn mehrere Personen wie gewohnt gemeinsam im Browser schreiben wollen.

### Einmaliges Setup durch die Person, die das Overleaf-Projekt erstellt

1. Auf [https://www.overleaf.com](https://www.overleaf.com) einloggen.
2. Neues Projekt erstellen.
3. Option `Blank Project` oder `Upload Project` verwenden.
4. Aus diesem Repository den Ordner `Docs/paper` vorbereiten.
5. In das Overleaf-Projekt hochladen:
   - `main.tex`
   - `references.bib`
   - kompletter Ordner `figures/`
6. In Overleaf sicherstellen, dass `main.tex` als Hauptdatei eingestellt ist.
7. Compiler auf `pdfLaTeX` stellen.
8. Projekt kompilieren.
9. Andere Autoren ueber `Share` einladen.

### Was nicht hochgeladen werden muss

Diese Dateien sind Build-Artefakte und sollten nicht als gemeinsame Arbeitsdateien behandelt werden:

- `main.aux`
- `main.bbl`
- `main.blg`
- `main.fdb_latexmk`
- `main.fls`
- `main.log`
- `main.pdf`
- `preview/`

Falls Overleaf beim ersten Build nach Literaturquellen fragt: normal neu kompilieren. Overleaf fuehrt BibTeX in der Regel automatisch aus, sobald `references.bib` eingebunden ist.

### Empfohlener Arbeitsablauf mit Overleaf

1. Alle schreiben im Overleaf-Projekt.
2. Die Person, die das GitHub-Repository pflegt, exportiert regelmaessig den Overleaf-Stand.
3. Im Repository werden dann mindestens diese Dateien aktualisiert:
   - `Docs/paper/main.tex`
   - `Docs/paper/references.bib`
   - neue oder geaenderte Dateien in `Docs/paper/figures/`
4. Automatische Dateien wie `.aux`, `.log`, `.bbl` und `.pdf` nur dann committen, wenn ihr das bewusst wollt. Fuer normale Zusammenarbeit sind sie nicht noetig.

### Optional: Overleaf mit Git verbinden

Overleaf kann je nach Plan mit Git genutzt werden. Das ist praktisch, aber nicht zwingend noetig.

Moegliche Varianten:

- Overleaf-Git direkt verwenden und das Projekt lokal klonen.
- Overleaf-GitHub-Sync verwenden, falls im Overleaf-Account verfuegbar.
- Einfacher manueller Weg: Overleaf-Projekt als ZIP herunterladen und die relevanten Dateien in `Docs/paper` ersetzen.

Fuer kleine Teams ist der manuelle Export oft weniger fehleranfaellig als ein halb eingerichteter Git-Sync.

## Variante B: Lokal in VS Code arbeiten

Diese Variante ist sinnvoll, wenn jemand ohne Overleaf direkt mit den Dateien im Repository arbeiten moechte.

### Benötigte Programme

Auf dem anderen Rechner muessen installiert sein:

1. Git
2. Visual Studio Code
3. Eine LaTeX-Distribution
4. VS-Code-Erweiterung `LaTeX Workshop`

### Windows

1. Git installieren:
   - [https://git-scm.com/download/win](https://git-scm.com/download/win)
2. Visual Studio Code installieren:
   - [https://code.visualstudio.com](https://code.visualstudio.com)
3. MiKTeX installieren:
   - [https://miktex.org/download](https://miktex.org/download)
4. MiKTeX Console oeffnen.
5. In MiKTeX einstellen, dass fehlende Pakete automatisch installiert werden duerfen.
6. In VS Code die Erweiterung `LaTeX Workshop` installieren.

### macOS

1. Git installieren, falls noch nicht vorhanden.
2. Visual Studio Code installieren:
   - [https://code.visualstudio.com](https://code.visualstudio.com)
3. MacTeX installieren:
   - [https://www.tug.org/mactex/](https://www.tug.org/mactex/)
4. In VS Code die Erweiterung `LaTeX Workshop` installieren.

### Linux

1. Git installieren.
2. Visual Studio Code installieren.
3. TeX Live installieren.
4. In VS Code die Erweiterung `LaTeX Workshop` installieren.

Beispiel fuer Ubuntu/Debian:

```bash
sudo apt update
sudo apt install git texlive-full latexmk
```

`texlive-full` ist gross, vermeidet aber fehlende LaTeX-Pakete.

## Repository lokal oeffnen

1. Repository klonen:

```bash
git clone <REPOSITORY-URL>
```

2. In den Projektordner wechseln:

```bash
cd MSAEES
```

3. Projekt in VS Code oeffnen:

```bash
code .
```

4. Datei oeffnen:

```text
Docs/paper/main.tex
```

5. In VS Code rechts oben oder ueber die LaTeX-Workshop-Seitenleiste `Build LaTeX project` ausfuehren.
6. Danach `View LaTeX PDF` oeffnen.

## Manuell lokal bauen

Falls der Build in VS Code nicht direkt klappt, kann im Terminal getestet werden:

```bash
cd Docs/paper
latexmk -pdf main.tex
```

Wenn `latexmk` nicht verfuegbar ist, geht auch:

```bash
cd Docs/paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Die mehrfachen LaTeX-Laeufe sind normal, weil Referenzen, Zitate und Abbildungsnummern erst ueber Hilfsdateien aufgeloest werden.

## Typische Probleme

### PDF zeigt keine Literaturquellen

Loesung:

1. Sicherstellen, dass `references.bib` im gleichen Ordner wie `main.tex` liegt.
2. Projekt nochmal vollstaendig bauen.
3. Bei manuellem Build `bibtex main` zwischen den `pdflatex`-Laeufen ausfuehren.

### Bilder fehlen

Loesung:

1. Pruefen, ob der Ordner `figures/` neben `main.tex` liegt.
2. Pruefen, ob Dateinamen exakt stimmen, inklusive Gross-/Kleinschreibung.
3. Keine Bilder aus anderen Ordnern referenzieren, solange das Paper in Overleaf laufen soll.

### Fehlendes Paket

Unter Windows in MiKTeX die automatische Paketinstallation aktivieren.

Unter Linux entweder fehlende Pakete nachinstallieren oder `texlive-full` verwenden.

### VS Code baut nicht automatisch

Loesung:

1. `LaTeX Workshop` installieren.
2. VS Code neu starten.
3. `Docs/paper/main.tex` oeffnen.
4. LaTeX-Workshop-Build erneut starten.

## Empfohlene Regeln fuer gemeinsame Arbeit

1. Immer nur an `Docs/paper/main.tex`, `Docs/paper/references.bib` und Dateien in `Docs/paper/figures/` arbeiten.
2. Keine automatisch erzeugten LaTeX-Dateien manuell bearbeiten.
3. Neue Abbildungen immer in `Docs/paper/figures/` ablegen.
4. Bildpfade immer relativ schreiben, zum Beispiel:

```latex
\includegraphics[width=\columnwidth]{figures/example.png}
```

5. Literatur nur in `references.bib` pflegen und im Text mit `\cite{...}` zitieren.
6. Vor dem Teilen oder Committen einmal frisch bauen und pruefen, ob die PDF ohne Fehler entsteht.

## Empfehlung fuer dieses Projekt

Fuer eure aktuelle Situation ist Overleaf wahrscheinlich der beste gemeinsame Schreibort. Der Aufwand ist gering, weil das Paper keine komplexe lokale Build-Umgebung braucht. Fuer lokale Mitarbeit reicht die VS-Code-Anleitung oben.

Wenn ihr Overleaf nutzt, sollte eine Person regelmaessig den sauberen Overleaf-Stand zurueck ins Repository uebernehmen, damit GitHub und Overleaf nicht auseinanderlaufen.
