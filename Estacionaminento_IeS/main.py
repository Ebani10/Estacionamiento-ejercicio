from Models.lugar import Lugar
from Models.vehiculo import Vehiculo
from services.estacionamiento_service import EstacionamientoService


def main():
	from Reconocedor_letras.Reconocedor import reconocer_placa

	lugares = [Lugar(1), Lugar(2), Lugar(3)]
	estacionamiento = EstacionamientoService(lugares)

	placa_esperada = input("Escribe la placa esperada (una letra): ").strip().upper()
	if len(placa_esperada) != 1 or not placa_esperada.isalpha():
		print("La placa debe contener exactamente una letra.")
		return

	tipo = input("Escribe el tipo de vehiculo: ").strip() or "Automovil"
	print("Iniciando reconocedor. Escribe la placa y presiona ESPACIO.")

	try:
		placa_reconocida = reconocer_placa()
	except RuntimeError as error:
		print(error)
		return

	if placa_reconocida is None:
		print("Reconocimiento cancelado.")
		return

	vehiculo = Vehiculo(placa_esperada, tipo)
	ticket = estacionamiento.registrar_entrada(vehiculo, placa_reconocida)
	if ticket is None:
		print("La placa reconocida no coincide con la placa esperada.")
		return

	ticket.mostrar()


if __name__ == "__main__":
	main()