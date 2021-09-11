from flask import Blueprint, jsonify

main =Blueprint('main', __name__)

@main.route('/process', methods = ['POST'])
def process():

    return 'DONE' , 201
