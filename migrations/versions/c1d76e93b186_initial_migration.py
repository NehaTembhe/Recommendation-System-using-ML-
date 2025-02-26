"""Initial migration.

Revision ID: c1d76e93b186
Revises: 
Create Date: 2025-01-22 13:46:19.900505

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'c1d76e93b186'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('signin', schema=None) as batch_op:
        batch_op.create_unique_constraint(None, ['username'])

    with op.batch_alter_table('signup', schema=None) as batch_op:
        batch_op.add_column(sa.Column('mobile', sa.String(length=15), nullable=False))
        batch_op.create_unique_constraint(None, ['username'])
        batch_op.create_unique_constraint(None, ['mobile'])
        batch_op.create_unique_constraint(None, ['email'])

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('signup', schema=None) as batch_op:
        batch_op.drop_constraint(None, type_='unique')
        batch_op.drop_constraint(None, type_='unique')
        batch_op.drop_constraint(None, type_='unique')
        batch_op.drop_column('mobile')

    with op.batch_alter_table('signin', schema=None) as batch_op:
        batch_op.drop_constraint(None, type_='unique')

    # ### end Alembic commands ###
