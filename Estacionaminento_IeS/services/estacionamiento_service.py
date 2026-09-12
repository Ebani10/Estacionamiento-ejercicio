from Models.ticket import Ticket


class EstacionamientoService:
    def __init__(self, lugares):
        self.lugares = lugares
        self.tickets = []

    def registrar_entrada(self, vehiculo, placa_reconocida):
        if vehiculo.placa != placa_reconocida:
            return None

        for lugar in self.lugares:
            if not lugar.ocupado:
                lugar.ocupar()
                ticket = Ticket(
                    len(self.tickets) + 1,
                    vehiculo,
                    lugar
                )
                self.tickets.append(ticket)
                return ticket

        return None

    def registrar_salida(self, ticket):
        if ticket in self.tickets:
            ticket.lugar.liberar()
            self.tickets.remove(ticket)
            return True
        return False