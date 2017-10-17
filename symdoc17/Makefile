all:
	sqrt.html
	t3d.html

clean:
	rm sqrt.md sqrt.html
	rm -rf __pycache__

sqrt.html: sqrt.py
	python sqrt.py --symdoc
	pandoc -t html -o sqrt.html --standalone --mathjax sqrt.md

t3d.html: t3d.py
	python t3d.py --symdoc
	pandoc -t html -o t3d.html --standalone --mathjax t3d.md

